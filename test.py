# Laboratory Project for CMSC 165 A.Y. 2023-2024
# Authors: 1. Nagano, Tyrone Frederik
#          2. Wallit, Zyrus Matthew
#          3. Sunga, Rafael

import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("data/yolov4.weights", "data/yolov4.cfg")
layer_names = net.getUnconnectedOutLayersNames()

# Load COCO class names
with open("data/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam/video input
    _, frame = cap.read()

    # Detect objects from frame
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(layer_names)

    confidences = []
    boxes = []
    # Process detections
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] in ["backpack", "handbag", "suitcase"]:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)

                # Rectangle coordinates
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append((x,y,w,h))
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indices:
        box = boxes[i]
        x,y,w,h = box
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        # label = f"{classes[class_id]}: {confidences[i]:.2f}"
        # color = (0,255,0)
        # cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # Draw rectangle and label
            # cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
            

    # Display output
    cv2.imshow("Object Detection", frame)



    if cv2.waitKey(1) == ord('q'):
        break
    
# Release webcam and close all the windows
cap.release()
cv2.destroyAllWindows()