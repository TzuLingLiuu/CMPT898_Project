import torch  # type: ignore
from torchvision.ops import nms  # type: ignore
from PIL import Image, ImageDraw  # type: ignore
import numpy as np  # type: ignore
import cv2  # type: ignore
from skimage import util, morphology, measure  # type: ignore
import os
import json
import base64
import torch# type: ignore
import io  # Needed for base64 encoding


model_path = '/student/ywa826/project/CMPT898/Project/YOLO_v5/yolov5/runs/train/yolov5_nodules/weights/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model = model.to("cuda")


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
    return cv2.LUT(image, table)


def extract_root_region(image):
    # Convert to grayscale and enhance contrast
    image_gray = util.img_as_float(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY))
    image_gray = cv2.equalizeHist((image_gray * 255).astype('uint8'))
    image_gray = adjust_gamma(image_gray, gamma=0.01)

    # Morphological operations
    struct_element = morphology.disk(3)
    mask = morphology.binary_opening(image_gray, struct_element)
    mask = morphology.binary_closing(mask, struct_element)

    # Label regions in the mask and find the largest connected component
    labeled = measure.label(mask)
    regions = measure.regionprops(labeled)
    if len(regions) == 0:
        return None
    largest_region = max(regions, key=lambda r: r.area)
    mask = (labeled == largest_region.label).astype(np.uint8)  # Only keep the largest region

    return mask


def apply_mask(image, mask):
    """Apply mask to an image to zero out non-root regions."""
    return Image.fromarray(cv2.bitwise_and(np.array(image), np.array(image), mask=mask))




def detect_and_save_json(image_path, crop_size, stride, model, json_output_path, conf_threshold=0.6):
    # Load the full image
    image = Image.open(image_path).convert("RGB")
    root_mask = extract_root_region(image)
    if root_mask is None:
        print(f"Root mask could not be generated for {image_path}.")
        return

    w, h = image.size
    all_detections = []

    for i in range(0, w - crop_size + 1, stride):
        for j in range(0, h - crop_size + 1, stride):
            crop = image.crop((i, j, i + crop_size, j + crop_size))
            crop_mask = root_mask[j:j + crop_size, i:i + crop_size]
            masked_crop = apply_mask(crop, crop_mask)
            results = model(masked_crop)

            for result in results.xywh[0]:
                x_center, y_center, width, height, confidence, cls = result.tolist()
                if confidence < conf_threshold:
                    continue
                abs_x_center = x_center + i
                abs_y_center = y_center + j
                abs_x1 = abs_x_center - width / 2
                abs_y1 = abs_y_center - height / 2
                abs_x2 = abs_x_center + width / 2
                abs_y2 = abs_y_center + height / 2

                all_detections.append({
                    "x1": abs_x1,
                    "y1": abs_y1,
                    "x2": abs_x2,
                    "y2": abs_y2,
                    "confidence": confidence,
                    "class": "Nodule"
                })

    # Non-Max Suppression (NMS)
    final_detections = []
    if all_detections:
        detections_tensor = torch.tensor([[d["x1"], d["y1"], d["x2"], d["y2"], d["confidence"]] for d in all_detections])
        boxes = detections_tensor[:, :4]
        scores = detections_tensor[:, 4]
        indices = nms(boxes, scores, iou_threshold=0.5)
        final_detections = [all_detections[i] for i in indices]

    # Convert image to base64
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")  # Save the image as PNG in memory
    image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')  # Encode to base64

    # Extract the image name from the path
    img_name = os.path.splitext(os.path.basename(image_path))[0]

    # LabelMe JSON format
    labelme_data = {
        "version": "5.2.1",
        "flags": {},
        "shapes": [],
        "imagePath": f"../Psudo_Labeled_Images/{img_name}.png",
        "imageData": image_data,  # Add the base64 image data here
        "imageHeight": h,
        "imageWidth": w
    }

    for detection in final_detections:
        shape = {
            "label": detection["class"],
            "points": [
                [detection["x1"], detection["y1"]],
                [detection["x2"], detection["y2"]]
            ],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        }
        labelme_data["shapes"].append(shape)

    # Save to JSON
    with open(json_output_path, 'w') as json_file:
        json.dump(labelme_data, json_file, indent=4)
    print(f"LabelMe JSON file saved at {json_output_path}")



# def detect_and_save_json(image_path, crop_size, stride, model, json_output_path, conf_threshold=0.6):
#     # Load and mask the full image
#     image = Image.open(image_path).convert("RGB")
#     root_mask = extract_root_region(image)
#     if root_mask is None:
#         print(f"Root mask could not be generated for {image_path}.")
#         return

#     w, h = image.size
#     all_detections = []

#     for i in range(0, w - crop_size + 1, stride):
#         for j in range(0, h - crop_size + 1, stride):
#             crop = image.crop((i, j, i + crop_size, j + crop_size))

#             # Apply the root mask to the crop
#             crop_mask = root_mask[j:j + crop_size, i:i + crop_size]
#             masked_crop = apply_mask(crop, crop_mask)

#             # Perform detection on the masked crop
#             results = model(masked_crop)

#             for result in results.xywh[0]:
#                 x_center, y_center, width, height, confidence, cls = result.tolist()
#                 if confidence < conf_threshold:
#                     continue  # Skip detections below the confidence threshold

#                 # Convert to absolute coordinates
#                 abs_x_center = x_center + i
#                 abs_y_center = y_center + j
#                 abs_x1 = abs_x_center - width / 2
#                 abs_y1 = abs_y_center - height / 2
#                 abs_x2 = abs_x_center + width / 2
#                 abs_y2 = abs_y_center + height / 2

#                 all_detections.append({
#                     "x1": abs_x1,
#                     "y1": abs_y1,
#                     "x2": abs_x2,
#                     "y2": abs_y2,
#                     "confidence": confidence,
#                     "class": "Nodule"
#                 })

    # Apply Non-Max Suppression (NMS)
    if all_detections:
        detections_tensor = torch.tensor([[d["x1"], d["y1"], d["x2"], d["y2"], d["confidence"]] for d in all_detections])
        boxes = detections_tensor[:, :4]  # [x1, y1, x2, y2]
        scores = detections_tensor[:, 4]  # confidence
        indices = nms(boxes, scores, iou_threshold=0.5)
        final_detections = [all_detections[i] for i in indices]

        # Save final detections to JSON
        with open(json_output_path, 'w') as json_file:
            json.dump({"detections": final_detections}, json_file, indent=4)
        print(f"JSON file saved at {json_output_path}")


# Batch processing
def batch_process_images(input_folder, output_folder, crop_size, stride, model, conf_threshold=0.6):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in sorted(os.listdir(input_folder)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_folder, filename)
            json_output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".json")

            print(f"Processing {image_path}...")
            detect_and_save_json(image_path, crop_size, stride, model, json_output_path, conf_threshold)


# Define paths and parameters
input_folder = "/student/ywa826/project/CMPT898/Project/YOLO_v5/Psudo_Labeled_Input_Images"
output_folder = "/student/ywa826/project/CMPT898/Project/YOLO_v5/Psudo_Labeled_Output_jsons"
crop_size = 128  # YOLOv5 default input size
stride = 64      # Stride for cropping

# Execute batch processing
batch_process_images(input_folder, output_folder, crop_size, stride, model, conf_threshold=0.5)
