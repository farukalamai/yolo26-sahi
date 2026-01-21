import gradio as gr
import spaces

import sahi.utils
from sahi import AutoDetectionModel
import sahi.predict
import sahi.slicing
from PIL import Image
import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO
import sys
import types

if 'huggingface_hub.utils._errors' not in sys.modules:
    mock_errors = types.ModuleType('_errors')
    mock_errors.RepositoryNotFoundError = Exception
    sys.modules['huggingface_hub.utils._errors'] = mock_errors

IMAGE_SIZE = 640

# Download sample images
sahi.utils.file.download_from_url(
    "https://user-images.githubusercontent.com/34196005/142730935-2ace3999-a47b-49bb-83e0-2bdd509f1c90.jpg",
    "apple_tree.jpg",
)
sahi.utils.file.download_from_url(
    "https://user-images.githubusercontent.com/34196005/142730936-1b397756-52e5-43be-a949-42ec0134d5d8.jpg",
    "highway.jpg",
)
sahi.utils.file.download_from_url(
    "https://user-images.githubusercontent.com/34196005/142742871-bf485f84-0355-43a3-be86-96b44e63c3a2.jpg",
    "highway2.jpg",
)
sahi.utils.file.download_from_url(
    "https://user-images.githubusercontent.com/34196005/142742872-1fefcc4d-d7e6-4c43-bbb7-6b5982f7e4ba.jpg",
    "highway3.jpg",
)

# Global model variable
model = None

# Model list
DETECTION_MODELS = ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"]
SEGMENTATION_MODELS = ["yolo26n-seg.pt", "yolo26s-seg.pt", "yolo26m-seg.pt", "yolo26l-seg.pt", "yolo26x-seg.pt"]
ALL_MODELS = DETECTION_MODELS + SEGMENTATION_MODELS


def load_yolo_model(model_name, confidence_threshold=0.5):
    global model
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_name,
        device=None,
        confidence_threshold=confidence_threshold,
        image_size=IMAGE_SIZE
    )
    return model


def sahi_predictions_to_sv_detections(object_predictions, image_shape, is_segmentation=False):
    """Convert SAHI predictions to Supervision Detections."""
    if not object_predictions:
        return sv.Detections.empty()
    
    xyxy = np.array([
        [pred.bbox.minx, pred.bbox.miny, pred.bbox.maxx, pred.bbox.maxy]
        for pred in object_predictions
    ], dtype=np.float32)
    confidence = np.array([pred.score.value for pred in object_predictions], dtype=np.float32)
    class_id = np.array([pred.category.id for pred in object_predictions], dtype=np.int32)
    class_names = [pred.category.name for pred in object_predictions]
    
    mask = None
    if is_segmentation:
        img_h, img_w = image_shape[:2]
        masks = []
        
        for pred in object_predictions:
            if pred.mask is not None and pred.mask.bool_mask is not None:
                m = pred.mask.bool_mask
                if isinstance(m, np.ndarray):
                    if m.shape != (img_h, img_w):
                        m_resized = cv2.resize(m.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                        masks.append(m_resized.astype(bool))
                    else:
                        masks.append(m.astype(bool))
                else:
                    masks.append(np.array(m, dtype=bool))
            else:
                masks.append(np.zeros((img_h, img_w), dtype=bool))
        
        if masks:
            mask = np.array(masks, dtype=bool)
    
    detections = sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
        mask=mask
    )
    detections.data = {"class_name": np.array(class_names)}
    
    return detections


def annotate_image_with_supervision(
    image: np.ndarray,
    detections: sv.Detections,
    is_segmentation: bool = False
) -> np.ndarray:
    """Annotate image using Supervision."""
    annotated_image = image.copy()
    
    if len(detections) == 0:
        return annotated_image
    
    box_annotator = sv.BoxAnnotator(thickness=2)
    
    if is_segmentation and detections.mask is not None:
        try:
            mask_annotator = sv.MaskAnnotator(opacity=0.4)
            annotated_image = mask_annotator.annotate(scene=annotated_image, detections=detections)
        except Exception as e:
            print(f"Mask annotation failed: {e}, skipping masks")
    
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
    
    labels = []
    for i in range(len(detections)):
        class_name = detections.data.get("class_name", ["unknown"] * len(detections))[i]
        if detections.confidence is not None:
            label = f"{class_name} {detections.confidence[i]:.2f}"
        else:
            label = class_name
        labels.append(label)
    
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=5)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    
    return annotated_image


@spaces.GPU(duration=60)
def sahi_yolo_inference(
    image,
    yolo_model_name,
    confidence_threshold,
    max_detections,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    postprocess_type="NMS",
    postprocess_match_metric="IOS",
    postprocess_match_threshold=0.3,
    postprocess_class_agnostic=True,
):
    load_yolo_model(yolo_model_name, confidence_threshold)

    image_width, image_height = image.size
    sliced_bboxes = sahi.slicing.get_slice_bboxes(
        image_height, image_width, slice_height, slice_width,
        False, overlap_height_ratio, overlap_width_ratio,
    )
    if len(sliced_bboxes) > 60:
        raise ValueError(f"{len(sliced_bboxes)} slices are too many. Try smaller slice size.")

    is_segmentation = "seg" in yolo_model_name.lower()
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Standard inference
    prediction_result_1 = sahi.predict.get_prediction(image=image, detection_model=model)
    if max_detections and len(prediction_result_1.object_prediction_list) > max_detections:
        prediction_result_1.object_prediction_list = sorted(
            prediction_result_1.object_prediction_list, key=lambda x: x.score.value, reverse=True
        )[:max_detections]

    detections_1 = sahi_predictions_to_sv_detections(
        prediction_result_1.object_prediction_list, image_bgr.shape, is_segmentation
    )
    annotated_1 = annotate_image_with_supervision(image_bgr, detections_1, is_segmentation)
    output_1 = Image.fromarray(cv2.cvtColor(annotated_1, cv2.COLOR_BGR2RGB))

    # Sliced inference
    prediction_result_2 = sahi.predict.get_sliced_prediction(
        image=image, detection_model=model,
        slice_height=int(slice_height), slice_width=int(slice_width),
        overlap_height_ratio=overlap_height_ratio, overlap_width_ratio=overlap_width_ratio,
        postprocess_type=postprocess_type, postprocess_match_metric=postprocess_match_metric,
        postprocess_match_threshold=postprocess_match_threshold, postprocess_class_agnostic=postprocess_class_agnostic,
    )
    if max_detections and len(prediction_result_2.object_prediction_list) > max_detections:
        prediction_result_2.object_prediction_list = sorted(
            prediction_result_2.object_prediction_list, key=lambda x: x.score.value, reverse=True
        )[:max_detections]

    detections_2 = sahi_predictions_to_sv_detections(
        prediction_result_2.object_prediction_list, image_bgr.shape, is_segmentation
    )
    annotated_2 = annotate_image_with_supervision(image_bgr, detections_2, is_segmentation)
    output_2 = Image.fromarray(cv2.cvtColor(annotated_2, cv2.COLOR_BGR2RGB))

    return output_1, output_2


with gr.Blocks(title="YOLO26 + SAHI") as app:
    gr.Markdown("# Small object detection & segmentation using YOLO26 + SAHI")

    with gr.Row():
        with gr.Column(scale=1):
            original_image_input = gr.Image(type="pil", label="Input Image")
            yolo_model_dropdown = gr.Dropdown(choices=ALL_MODELS, value="yolo26s.pt", label="Model")
            
            with gr.Accordion("Detection Settings", open=True):
                confidence_threshold_slider = gr.Slider(0.0, 1.0, 0.01, value=0.5, label="Confidence Threshold")
                max_detections_slider = gr.Slider(1, 500, 1, value=300, label="Max Detections")
            
            with gr.Accordion("Slicing", open=False):
                slice_height_input = gr.Number(value=512, label="Slice Height")
                slice_width_input = gr.Number(value=512, label="Slice Width")
                overlap_height_ratio_slider = gr.Slider(0.0, 1.0, 0.01, value=0.2, label="Overlap Height Ratio")
                overlap_width_ratio_slider = gr.Slider(0.0, 1.0, 0.01, value=0.2, label="Overlap Width Ratio")
            
            with gr.Accordion("Postprocessing", open=False):
                postprocess_type_dropdown = gr.Dropdown(["NMS", "GREEDYNMM"], value="NMS", label="Type")
                postprocess_match_metric_dropdown = gr.Dropdown(["IOU", "IOS"], value="IOS", label="Match Metric")
                postprocess_match_threshold_slider = gr.Slider(0.0, 1.0, 0.01, value=0.3, label="Match Threshold")
                postprocess_class_agnostic_checkbox = gr.Checkbox(value=True, label="Class Agnostic")

            submit_button = gr.Button("Run Inference", variant="primary")

        with gr.Column(scale=1):
            output_standard = gr.Image(type="pil", label="Standard Inference")
            output_sahi_sliced = gr.Image(type="pil", label="SAHI Sliced Inference")

    gr.Examples(
        examples=[
            ["apple_tree.jpg", "yolo26s.pt", 0.5, 300, 256, 256, 0.2, 0.2, "NMS", "IOS", 0.3, True],
            ["highway.jpg", "yolo26s.pt", 0.5, 300, 256, 256, 0.2, 0.2, "NMS", "IOS", 0.3, True],
            ["highway2.jpg", "yolo26s.pt", 0.5, 300, 512, 512, 0.2, 0.2, "NMS", "IOS", 0.3, True],
            ["highway3.jpg", "yolo26s-seg.pt", 0.5, 300, 512, 512, 0.2, 0.2, "NMS", "IOS", 0.3, True],
        ],
        inputs=[
            original_image_input, yolo_model_dropdown, confidence_threshold_slider, max_detections_slider,
            slice_height_input, slice_width_input, overlap_height_ratio_slider, overlap_width_ratio_slider,
            postprocess_type_dropdown, postprocess_match_metric_dropdown, postprocess_match_threshold_slider,
            postprocess_class_agnostic_checkbox,
        ],
        outputs=[output_standard, output_sahi_sliced],
        fn=sahi_yolo_inference,
        cache_examples=True,
    )

    submit_button.click(
        fn=sahi_yolo_inference,
        inputs=[
            original_image_input, yolo_model_dropdown, confidence_threshold_slider, max_detections_slider,
            slice_height_input, slice_width_input, overlap_height_ratio_slider, overlap_width_ratio_slider,
            postprocess_type_dropdown, postprocess_match_metric_dropdown, postprocess_match_threshold_slider,
            postprocess_class_agnostic_checkbox,
        ],
        outputs=[output_standard, output_sahi_sliced],
    )

app.launch(mcp_server=True)