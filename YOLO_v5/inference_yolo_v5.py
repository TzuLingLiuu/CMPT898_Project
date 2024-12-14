import torch# type: ignore
from torchvision.ops import nms# type: ignore
from PIL import Image, ImageDraw# type: ignore
import numpy as np# type: ignore
import cv2# type: ignore
from skimage import util, morphology, measure# type: ignore
import matplotlib.pyplot as plt# type: ignore
from PIL import ImageFont# type: ignore


model_path = '/student/ywa826/project/CMPT898/Project/YOLO_v5/yolov5/runs/train/yolov5_nodules_exp/weights/best.pt'
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
    
    image_np = np.array(image)  # Convert PIL image to numpy array
    mask_3d = np.stack([mask] * 3, axis=-1)  # Expand mask to 3 channels
    masked_image = image_np * mask_3d  # Apply mask to each channel
    return Image.fromarray(masked_image.astype('uint8'))


def detect_and_restore(image_path, crop_size, stride, model, output_path="output.jpg", conf_threshold=0.6):
    # Load and mask the full image
    image = Image.open(image_path).convert("RGB")
    root_mask = extract_root_region(image)
    if root_mask is None:
        print("Root mask could not be generated.")
        return

    w, h = image.size
    all_detections = []

    for i in range(0, w - crop_size + 1, stride):
        for j in range(0, h - crop_size + 1, stride):
            crop = image.crop((i, j, i + crop_size, j + crop_size))

            # Apply the root mask to the crop
            crop_mask = root_mask[j:j + crop_size, i:i + crop_size]
            masked_crop = apply_mask(crop, crop_mask)

            # Perform detection on the masked crop
            results = model(masked_crop)

            for result in results.xywh[0]:
                x_center, y_center, width, height, confidence, cls = result.tolist()
                if confidence < conf_threshold:
                    continue  # Skip detections below the confidence threshold

                # Convert to absolute coordinates
                abs_x_center = x_center + i
                abs_y_center = y_center + j
                abs_x1 = abs_x_center - width / 2
                abs_y1 = abs_y_center - height / 2
                abs_x2 = abs_x_center + width / 2
                abs_y2 = abs_y_center + height / 2

                all_detections.append([abs_x1, abs_y1, abs_x2, abs_y2, confidence, cls])

    # Apply Non-Max Suppression (NMS)
    if all_detections:
        detections_tensor = torch.tensor(all_detections)
        boxes = detections_tensor[:, :4]  # [x1, y1, x2, y2]
        scores = detections_tensor[:, 4]  # confidence
        indices = nms(boxes, scores, iou_threshold=0.5)
        final_detections = [all_detections[i] for i in indices]
        #count how many final_detections
        font_size = 20
    
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        

        # Draw bounding boxes on the original image
        draw = ImageDraw.Draw(image)
        for det in final_detections:
            x1, y1, x2, y2, confidence, cls = det
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1), f"{confidence:.1f}", fill="red", font=font)

        # Save image with bounding boxes
        image.save(output_path)
        print(f"Image saved with bounding boxes at {output_path}")

# Call the detect_and_restore function
test_image_path = "/student/ywa826/project/CMPT898/Project/YOLO_v5/YOLODataset_Test/images/test/IMG_7165.png"
output_image_path = "/student/ywa826/project/CMPT898/Project/YOLO_v5/IMG_7165_Test_0.5_Output.png"
crop_size = 128  # YOLOv5 default input size
stride = 64      # Stride for cropping

# Execute detection and save the output image
detect_and_restore(test_image_path, crop_size, stride, model, output_image_path, 0.5)
