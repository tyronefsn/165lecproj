# Laboratory Project for CMSC 165 A.Y. 2023-2024
# Authors: 1. Nagano, Tyrone Frederik
#          2. Wallit, Zyrus Matthew
#          3. Sunga, Rafael

import cv2
import numpy as np


# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load YOLO
net = cv2.dnn.readNet("data/yolov4.weights", "data/yolov4.cfg")
layer_names = net.getUnconnectedOutLayersNames()

# Load COCO class names
with open("data/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

while True:
    ret, img = cap.read()
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416,416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(layer_names)

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]


            center_x = int(obj[0] * width)
            center_y = int(obj[1] * height)
            w = int(obj[2] * width)
            h = int(obj[3] * height)

            # Rectangle coordinates
            x = int(center_x - w/2)
            y = int(center_y - h/2)
            if confidence > 0.5:
              color = (0, 255, 0)
              cv2.rectangle(img, (center_x - w//2, center_y - h//2), (center_x + w//2, center_y + h//2), color, 2)
              label = f"{classes[class_id]}: {confidence:.2f}"
              cv2.putText(img, label, (center_x - w//2, center_y - h//2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Webcam", img)


    cv2.waitKey(1)

