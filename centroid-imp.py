import datetime
import cv2
import pygame
from pygame.locals import *
from tkinter import Tk, filedialog
from PIL import Image
from io import BytesIO
import numpy as np
# https://github.com/mailrocketsystems/AIComputerVision/tree/master
from centroidtracker import CentroidTracker
class BagAlertApp:
    def __init__(self):
        self.resolution = (1200, 700)
        self.main_interface()

    def main_interface(self):
        pygame.init()
        self.screen = pygame.display.set_mode(self.resolution)
        pygame.display.set_caption("Bag Alert")

        font = pygame.font.Font(None, 36)

        video_sample_button = pygame.Rect(400, 200, 400, 100)
        live_feed_button = pygame.Rect(400, 350, 400, 100)

        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    return
                elif event.type == MOUSEBUTTONDOWN:
                    if video_sample_button.collidepoint(event.pos):
                        pygame.quit()
                        self.show_video_sample()
                        self.main_interface()
                    elif live_feed_button.collidepoint(event.pos):
                        pygame.quit()
                        self.show_live_feed()
                        self.main_interface()

            self.screen.fill((255, 255, 255))

            pygame.draw.rect(self.screen, (0, 128, 255), video_sample_button)
            pygame.draw.rect(self.screen, (0, 128, 255), live_feed_button)

            text = font.render("Video Sample", True, (255, 255, 255))
            self.screen.blit(text, (500, 240))

            text = font.render("Live Feed", True, (255, 255, 255))
            self.screen.blit(text, (530, 390))

            pygame.display.flip()

    def show_video_sample(self):
        root = Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
        cap = cv2.VideoCapture(file_path)
        self.display_video(cap)

    def show_live_feed(self):
        net = cv2.dnn.readNet("data/yolov4.weights", "data/yolov4.cfg")
        layer_names = net.getUnconnectedOutLayersNames()

        # Load COCO class names
        with open("data/coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]

        # Initialize webcam
        file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
        cap = cv2.VideoCapture(file_path)
        fps_start_time = datetime.datetime.now()
        fps = 0
        total_frames = 0
        # tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)
        tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)
        prev_box = []
        prev_conf = []
        while True:
            # Read frame from webcam/video input
            ret, frame = cap.read()
            if not ret:
                continue
            total_frames+=1
            # try:
            #     success, new_bbox = tracker.update(frame)
            #     if success:
            #         for i, newbox in enumerate(new_bbox):
            #             print(newbox)
            #             boxes.append(newbox)
            #             confidences.append(1.0)
            # except Exception as e:
            #     print(e)
            
            # Detect objects from frame
            height, width, _ = frame.shape
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB=True, crop=False)
            net.setInput(blob)
            detections = net.forward(layer_names)

            confidences = []
            labels = []
            boxes = []
            # Process detections
            for detection in detections:
                for obj in detection:
                    scores = obj[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.3 and classes[class_id] in ["backpack", "handbag", "suitcase"]:
                        center_x = int(obj[0] * width)
                        center_y = int(obj[1] * height)
                        w = int(obj[2] * width)
                        h = int(obj[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w/2)
                        y = int(center_y - h/2)
                        boxes.append((x,y,w,h))
                        confidences.append(float(confidence))

            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)

            fps_end_time = datetime.datetime.now()
            time_diff = fps_end_time - fps_start_time
            if time_diff.seconds == 0:
                fps = 0.0
            else:
                fps = (total_frames / time_diff.seconds)

            fps_text = "FPS: {:.2f}".format(fps)
            tfps_text = "Total Frames: {:.2f}".format(total_frames)

            cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            cv2.putText(frame, tfps_text, (5, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

            bboxes = []
            for i in indices:
                bboxes.append(boxes[i])
            print(bboxes)
            objects = tracker.update(bboxes)
            for (objectId, bbox) in objects.items():
                x1, y1, x2, y2 = bbox
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                cv2.rectangle(frame, (x1,y1), (x1+x2, y1+y2), (0,255,0), 2)
                text = "ID: {}".format(objectId)
                cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            # for i in indices:
            #     box = boxes[i]
            #     x,y,w,h = box
            #     x = int(x)
            #     y = int(y)
            #     w = int(w)
            #     h = int(h)
                # cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                # tracker.add(cv2.legacy.TrackerCSRT.create(), frame, box)
                # cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                # success, new_bbox = tracker.update(frame)
                # if success:
                #     new_bbox = tuple(map(int, new_bbox))
                #     if all(coord > 0 for coord in new_bbox):
                #         cv2.rectangle(frame, new_bbox[:2], (new_bbox[0] + new_bbox[2], new_bbox[1] + new_bbox[3]), (0, 255, 0), 2)
                #         prev_box.append(new_bbox)
                #         prev_conf.append(confidences[i])
                    

            # Display output
            cv2.imshow("Object Detection", frame)



            if cv2.waitKey(1) == ord('q'):
                break
            
        # Release webcam and close all the windows
        cap.release()
        cv2.destroyAllWindows()

    def display_video(self, cap):
        pygame.init()
        screen = pygame.display.set_mode(self.resolution)
        pygame.display.set_caption("Video Sample")

        clock = pygame.time.Clock()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, self.resolution)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            image = pygame.image.fromstring(image.tobytes(), image.size, image.mode)

            screen.blit(image, (0, 0))
            pygame.display.flip()

            clock.tick(30)

            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    cap.release()
                    return

        cap.release()


if __name__ == "__main__":
    app = BagAlertApp()
