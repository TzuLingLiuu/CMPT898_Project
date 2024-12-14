from ultralytics import RTDETR # type: ignore

# Load a COCO-pretrained RT-DETR-l model
model = RTDETR("rtdetr-l.pt")

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="/student/ywa826/project/CMPT898/Project/RTDETR/YOLODataset/dataset.yaml", epochs=20, imgsz=128)
model = model.to("cuda")
# Run inference with the RT-DETR-l model on the 'bus.jpg' image
results = model("/student/ywa826/project/CMPT898/Project/YOLO_v5/IMG_3381.png")
