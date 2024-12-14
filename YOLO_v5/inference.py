import os
import json
import torch# type: ignore
from torchvision.ops import nms# type: ignore
from PIL import Image, ImageDraw# type: ignore
import numpy as np# type: ignore
import cv2# type: ignore
import matplotlib.pyplot as plt# type: ignore
from PIL import ImageFont# type: ignore 
from pathlib import Path
from skimage import io, util, morphology, measure, filters, feature# type: ignore

# Load the YOLOv5 model
model_path = '/student/ywa826/project/CMPT898/Project/YOLO_v5/yolov5/runs/train/yolov5_nodules/weights/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model = model.to("cuda")

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
    return cv2.LUT(image, table)

def enhance_details(image_gray):

    gaussian_3 = cv2.GaussianBlur(image_gray, (0, 0), 2.0)
    unsharp_image = cv2.addWeighted(image_gray, 1.5, gaussian_3, -0.5, 0)
    return unsharp_image

def adaptive_threshold(image_gray):

 
    global_thresh = filters.threshold_otsu(image_gray)
    local_thresh = filters.threshold_local(image_gray, block_size=35, method='gaussian')

    global_mask = image_gray > global_thresh
    local_mask = image_gray > local_thresh
    
    combined_mask = global_mask | local_mask
    return combined_mask

def extract_root_region(image):
  
  
    image_gray = util.img_as_float(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY))
    
    
    image_gray[:180, :] = 0  
    image_gray[-180:, :] = 0  
    image_gray[:500, :500] = 0  
    image_gray[:500, -500:] = 0  
    

    image_gray = cv2.equalizeHist((image_gray * 255).astype('uint8'))
    image_gray = adjust_gamma(image_gray, gamma=0.03)
    image_gray = enhance_details(image_gray)
    
    edges = feature.canny(image_gray, sigma=1.5, low_threshold=0.05, high_threshold=0.15)
    
   
    binary_mask = adaptive_threshold(image_gray)
    
   
    small_struct = morphology.disk(0.7)
    large_struct = morphology.disk(1.2)
    
  
    mask_fine = morphology.binary_opening(binary_mask, small_struct)
    mask_fine = morphology.binary_closing(mask_fine, small_struct)

    mask_coarse = morphology.binary_opening(binary_mask, large_struct)
    mask_coarse = morphology.binary_closing(mask_coarse, large_struct)
    
   
    mask_combined = mask_fine | mask_coarse
    
    mask_final = morphology.remove_small_objects(mask_combined, min_size=50, connectivity=2)
    
  
    skeleton = morphology.skeletonize(mask_final)
  
    dilated_skeleton = morphology.binary_dilation(skeleton, morphology.disk(0.8))
    
  
    final_mask = mask_final | dilated_skeleton
    
   
    labeled = measure.label(final_mask)
    regions = measure.regionprops(labeled)
    print(f"Number of regions found: {len(regions)}")
    
    if regions:
      
        largest_region = max(regions, key=lambda r: r.area)
        final_result =  (labeled == largest_region.label).astype(np.uint8)  # Only keep the largest region
        return final_result


def apply_mask(image, mask):
    
    image_np = np.array(image)  # Convert PIL image to numpy array
    mask_3d = np.stack([mask] * 3, axis=-1)  # Expand mask to 3 channels
    masked_image = image_np * mask_3d  # Apply mask to each channel
    return Image.fromarray(masked_image.astype('uint8'))


def detect_and_restore_batch(test_folder, output_folder_images, output_folder_labels, model, crop_size, stride, conf_threshold=0.6):
    os.makedirs(output_folder_images, exist_ok=True)
    os.makedirs(output_folder_labels, exist_ok=True)

    def calculate_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0

    def filter_overlapping_boxes(detections, overlap_threshold=0.7, max_overlap_count=3):
       
        filtered_detections = []
        
        for i, det in enumerate(detections):
            overlap_count = 0
            for j, other_det in enumerate(detections):
                if i != j:
                    iou = calculate_iou(det[:4], other_det[:4])
                    if iou > overlap_threshold:
                        overlap_count += 1
                        
                        # If this detection has lower confidence than the other one,
                        # skip it and break the inner loop
                        if det[4] < other_det[4]:
                            break
            
            # Only add the detection if it doesn't have too many overlaps
            # or if it has the highest confidence among its overlapping group
            if overlap_count < max_overlap_count:
                filtered_detections.append(det)
                
        return filtered_detections

    for image_file in Path(test_folder).glob("*.png"):
        image_path = str(image_file)
        image = Image.open(image_path).convert("RGB")
        root_mask = extract_root_region(image)

        if root_mask is None:
            print(f"Root mask could not be generated for {image_path}.")
            continue

        w, h = image.size
        all_detections = []

        for i in range(0, w - crop_size + 1, stride):
            for j in range(0, h - crop_size + 1, stride):
                crop = image.crop((i, j, i + crop_size, j + crop_size))
                crop_mask = root_mask[j:j + crop_size, i:i + crop_size]
                masked_crop = apply_mask(crop, crop_mask)
                model_input = np.array(masked_crop)
                results = model(model_input)

                for result in results.xywh[0]:
                    x_center, y_center, width, height, confidence, cls = result.tolist()
                    if confidence < conf_threshold:
                        continue

                    # Convert from crop-local to global image coordinates
                    abs_x_center = x_center + i
                    abs_y_center = y_center + j
                    abs_x1 = abs_x_center - width / 2
                    abs_y1 = abs_y_center - height / 2
                    abs_x2 = abs_x_center + width / 2
                    abs_y2 = abs_y_center + height / 2

                    all_detections.append([abs_x1, abs_y1, abs_x2, abs_y2, confidence, int(cls)])

        if all_detections:
            # First apply NMS
            detections_tensor = torch.tensor(all_detections)
            boxes = detections_tensor[:, :4]
            scores = detections_tensor[:, 4]
            indices = nms(boxes, scores, iou_threshold=0.5)
            nms_detections = [all_detections[i] for i in indices]
            
            # Then filter overlapping boxes
            final_detections = filter_overlapping_boxes(nms_detections)

            # Draw detections and prepare YOLO format outputs
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype("DejaVuSans.ttf", 20)

            yolo_lines = []
            for det in final_detections:
                x1, y1, x2, y2, confidence, cls = det

                # Draw bounding boxes
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                draw.text((x1, y1), f"{confidence:.2f}", fill="red", font=font)

                # YOLO format normalization
                x_center = (x1 + x2) / 2 / w
                y_center = (y1 + y2) / 2 / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h

                yolo_lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            output_image_path = os.path.join(output_folder_images, os.path.basename(image_path))
            image.save(output_image_path)
            print(f"Image saved with bounding boxes at {output_image_path}")

            txt_filename = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
            txt_path = os.path.join(output_folder_labels, txt_filename)
            with open(txt_path, "w") as f:
                f.write("\n".join(yolo_lines))
            print(f"YOLO format detection results saved to {txt_path}")

# Paths and parameters
test_folder = "/student/ywa826/project/CMPT898/Project/YOLO_v5/YOLODataset_Test/images/test"
output_folder_images= "/student/ywa826/project/CMPT898/Project/YOLO_v5/test_outputs/images"
output_folder_labels= "/student/ywa826/project/CMPT898/Project/YOLO_v5/test_outputs/labels"
crop_size = 128
stride = 64

# Execute batch detection 
detect_and_restore_batch(test_folder, output_folder_images, output_folder_labels, model, crop_size, stride, 0.6)

