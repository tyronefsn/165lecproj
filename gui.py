from tkinter import *
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import datetime

import numpy as np
from centroidtracker import CentroidTracker
def main():
    root = Tk()
    root.title("CMSC 165 Project")
    AppGUI(root)
    root.mainloop()

class AppGUI():
    def __init__(self, root):
        self.appFrame = Frame(root)
        self.appFrame.pack()

        self.video_source = None
        self.video_capture = cv2.VideoCapture(self.video_source)

        self.createVideoMenu()
        self.createVideoArea()
        self.createMenu()

        self.playing = False

    def createVideoMenu(self):
        self.videoMenu = Frame(self.appFrame)
        self.videoMenu.pack()

        self.openVideoBtn = Button(self.videoMenu, text="Open Video", font=("Helvetica", 15), height=1, width=10, bg="silver", command=self.openVideo)
        self.openVideoBtn.grid(row=0, column=0)

        self.playVideoBtn = Button(self.videoMenu, text="Play", font=("Helvetica", 15), height=1, width=10, bg="silver", command=self.play_video)
        self.playVideoBtn.grid(row=0, column=1)

        self.pauseVideoBtn = Button(self.videoMenu, text="Pause", font=("Helvetica", 15), height=1, width=10, bg="silver", command=self.stop_video)
        self.pauseVideoBtn.grid(row=0, column=2)

    def createVideoArea(self):
        
        self.videoArea = Canvas(self.appFrame, width=self.video_capture.get(3), height=self.video_capture.get(4))
        self.videoArea.pack()

    def createMenu(self):

        self.setTimeFrame = Frame(self.appFrame)
        self.setTimeFrame.pack()

        self.hourLabel = Label(self.setTimeFrame, font =("Helvetica", 15), text = "Hour")
        self.hourLabel.grid(row=0, column=0)

        self.setHour = Spinbox(self.setTimeFrame, from_=0, to=24)
        self.setHour.grid(row=1, column=0)

        self.minuteLabel = Label(self.setTimeFrame, font =("Helvetica", 15), text = "Minute")
        self.minuteLabel.grid(row=0, column=1)

        self.setMinute = Spinbox(self.setTimeFrame, from_=0, to=59)
        self.setMinute.grid(row=1, column=1)

        self.secondLabel = Label(self.setTimeFrame, font =("Helvetica", 15), text = "Second")
        self.secondLabel.grid(row=0, column=2)

        self.setSecond = Spinbox(self.setTimeFrame, from_=0, to=59)
        self.setSecond.grid(row=1, column=2)

        self.setTimeBtn = Button(self.appFrame, text="Set Time", font=("Helvetica", 15), height=1, width=10, bg="silver")
        self.setTimeBtn.pack()

    def play_video(self):
        net = cv2.dnn.readNet("data/yolov4.weights", "data/yolov4.cfg")
        layer_names = net.getUnconnectedOutLayersNames()

        # Load COCO class names
        with open("data/coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]

        # Initialize webcam
        file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
        self.video_source = file_path
        self.video_capture = cv2.VideoCapture(self.video_source)
        fps_start_time = datetime.datetime.now()
        fps = 0
        total_frames = 0
        # tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)
        tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)
        prev_box = []
        prev_conf = []


        self.playing = True
        if self.video_capture:
            self.playing = True
            while self.playing:
                ret, frame = self.video_capture.read()
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

                self.display_frame(frame)
                self.appFrame.update_idletasks()
                self.appFrame.update()

    def stop_video(self):
        self.playing = False


    def openVideo(self):
        file_path = filedialog.askopenfilename(initialdir='../', title="Select Video File", filetypes=(("Video Files", "*.mp4"), ("all files", "*.*")))

        if file_path:
            self.video_capture.release()
            self.video_capture = cv2.VideoCapture(file_path)

            ret, frame = self.video_capture.read()
            if ret:
                self.display_frame(frame)
 
    def display_frame(self, frame):
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(img)

            self.videoArea.config(width=img.width(), height=img.height())
            self.videoArea.create_image(0, 0, anchor=NW, image=img)
            self.videoArea.image = img

main()