import cv2 # type: ignore
import torch# type: ignore
from PIL import Image# type: ignore

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s") #, autoshape=False , autoshape=False
model = model.to("cuda")
# Images
for f in "zidane.jpg", "bus.jpg":
    torch.hub.download_url_to_file("https://ultralytics.com/images/" + f, f)  # download 2 images
im1 = Image.open("zidane.jpg")  # PIL image
im2 = cv2.imread("bus.jpg")[..., ::-1]  # OpenCV image (BGR to RGB)

# Inference
results = model([im1, im2], size=640)  # batch of images

# Results
results.print()
results.save()  # or .show()

