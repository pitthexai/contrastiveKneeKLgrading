from PIL import Image
import cv2

image_path = "C:/kl_grading/yolov11/datasets/xray_datasets/yolo_datasets/full_images/9000099.jpg"

# image_path = "C:/kl_grading/yolov11/datasets/xray_datasets/yolo_datasets/cropped_images/9000099_l.jpg"
image = Image.open(image_path)

print(image.mode)

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
print(img.shape)