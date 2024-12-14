import torch# type: ignore
from torchvision.ops import nms# type: ignore
from PIL import Image, ImageDraw# type: ignore
import numpy as np# type: ignore

# 載入模型
model_path = '/home/tzulingliu/Documents/CMPT898/Project/YOLO_v5/yolov5/runs/train/yolov5_nodules/weights/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model = model.to("cuda")

def detect_and_restore(image_path, crop_size, stride, model, output_path="output.jpg"):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    all_detections = []

    for i in range(0, w - crop_size + 1, stride):
        for j in range(0, h - crop_size + 1, stride):
            crop = image.crop((i, j, i + crop_size, j + crop_size))
            results = model(crop)

            for result in results.xywh[0]:
                x_center, y_center, width, height, confidence, cls = result.tolist()

                # 將寬度和高度轉換為左上角和右下角的座標
                abs_x_center = x_center + i
                abs_y_center = y_center + j
                abs_x1 = abs_x_center - width / 2
                abs_y1 = abs_y_center - height / 2
                abs_x2 = abs_x_center + width / 2
                abs_y2 = abs_y_center + height / 2

                # 儲存還原後的坐標和其他資訊
                all_detections.append([abs_x1, abs_y1, abs_x2, abs_y2, confidence, cls])

    # 非極大值抑制 (NMS)
    if all_detections:
        detections_tensor = torch.tensor(all_detections)
        boxes = detections_tensor[:, :4]  # [x1, y1, x2, y2]
        scores = detections_tensor[:, 4]  # confidence
        indices = nms(boxes, scores, iou_threshold=0.5)
        final_detections = [all_detections[i] for i in indices]

        # 繪製bounding boxes在原圖上
        draw = ImageDraw.Draw(image)
        for det in final_detections:
            x1, y1, x2, y2, confidence, cls = det
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1), f"{int(cls)}: {confidence:.2f}", fill="red")

        # 儲存包含bounding boxes的圖像
        image.save(output_path)
        print(f"Image saved with bounding boxes at {output_path}")

# 呼叫detect_and_restore以進行測試並儲存結果
test_image_path = "/home/tzulingliu/Documents/CMPT898/Project/YOLO_v5/IMG_3381.png"
output_image_path = "/home/tzulingliu/Documents/CMPT898/Project/YOLO_v5/IMG_3381_Test_Output.png"
crop_size = 128 # YOLOv5默認的輸入大小
stride = 64     # 可調整的步幅，越小表示重疊越多

# 執行檢測並儲存圖片
detect_and_restore(test_image_path, crop_size, stride, model, output_image_path)
