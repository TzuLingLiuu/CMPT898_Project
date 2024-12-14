import os
import cv2# type: ignore

input_image_dir = '/student/ywa826/project/CMPT898/Project/YOLO_v5/YOLODataset/origin_images'
input_train_image_dir = os.path.join(input_image_dir, 'train')
input_valid_image_dir = os.path.join(input_image_dir, 'val')

input_label_dir = '/student/ywa826/project/CMPT898/Project/YOLO_v5/YOLODataset/origin_labels'
input_label_train_dir = os.path.join(input_label_dir, 'train')
input_label_valid_dir = os.path.join(input_label_dir, 'val')

output_image_dir = "/student/ywa826/project/CMPT898/Project/YOLO_v5/YOLODataset/images"
output_label_dir = "/student/ywa826/project/CMPT898/Project/YOLO_v5/YOLODataset/labels"
output_train_image_dir = os.path.join(output_image_dir, 'train')
output_valid_image_dir = os.path.join(output_image_dir, 'val')
output_train_label_dir = os.path.join(output_label_dir, 'train')
output_valid_label_dir = os.path.join(output_label_dir, 'val')

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)
os.makedirs(output_train_image_dir, exist_ok=True)
os.makedirs(output_valid_image_dir, exist_ok=True)
os.makedirs(output_train_label_dir, exist_ok=True)
os.makedirs(output_valid_label_dir, exist_ok=True)

crop_size = 128  
stride = 64     # crop_size 的 50%


def crop_and_save(image_path, label_path, output_image_dir, output_label_dir, crop_size, stride):
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    filename = os.path.splitext(os.path.basename(image_path))[0]

    # 讀取標籤文件
    if os.path.exists(label_path):
        with open(label_path, 'r') as file:
            labels = file.readlines()
    else:
        labels = []

    # 使用滑動窗口切割圖片
    for i in range(0, w - crop_size + 1, stride):
        for j in range(0, h - crop_size + 1, stride):
            crop = image[j:j+crop_size, i:i+crop_size]
            if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
                continue

            # 檢查小塊中是否有標註的物體
            valid_labels = []
            for line in labels:
                cls, x, y, width, height = map(float, line.strip().split())
                x_center = x * w
                y_center = y * h
                box_width = width * w
                box_height = height * h

                # 將bounding box平移到裁剪區域內的座標
                x_center -= i
                y_center -= j

                # 確保物體中心位於crop範圍內
                if 0 <= x_center <= crop_size and 0 <= y_center <= crop_size:
                    # 根據裁剪區域重新計算bounding box比例
                    x_center /= crop_size
                    y_center /= crop_size
                    box_width /= crop_size
                    box_height /= crop_size
                    valid_labels.append(f"{int(cls)} {x_center} {y_center} {box_width} {box_height}")

            # 如果小塊包含物體，才保存小圖和標註
            if valid_labels:
                crop_filename = f"{filename}_{j}_{i}.png"
                cv2.imwrite(os.path.join(output_image_dir, crop_filename), crop)
                with open(os.path.join(output_label_dir, crop_filename.replace('.png', '.txt')), 'w') as label_out:
                    label_out.write("\n".join(valid_labels))

# 執行批次處理
for image_file in os.listdir(input_train_image_dir):
    image_path = os.path.join(input_train_image_dir, image_file)
    label_path = os.path.join(input_label_train_dir, image_file.replace('.png', '.txt'))
    crop_and_save(image_path, label_path, output_train_image_dir, output_train_label_dir, crop_size, stride)

for image_file in os.listdir(input_valid_image_dir):
    image_path = os.path.join(input_valid_image_dir, image_file)
    label_path = os.path.join(input_label_valid_dir, image_file.replace('.png', '.txt'))
    crop_and_save(image_path, label_path, output_valid_image_dir, output_valid_label_dir, crop_size, stride)
