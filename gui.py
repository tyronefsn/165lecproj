from tkinter import *
import tkinter.messagebox as messagebox
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import datetime
import threading

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
        # Init video_capture to None to avoid errors 
        self.video_capture = None
        self.timeTreshold = 0

        self.playing = False

        self.hour = 0
        self.distance = 0
        self.frame = 0
        
        self.createVideoMenu()
        self.createVideoArea()
        self.createMenu()
        # Added a protocol to release video capture after
        root.protocol("WM_DELETE_WINDOW", lambda: [self.on_window_close(),  root.destroy(), root.quit()])

    def on_window_close(self):
        # Close the window
        if self.video_capture == None:
            return
        self.video_capture.release()

    def createVideoMenu(self):
        self.videoMenu = Frame(self.appFrame)
        self.videoMenu.pack()


        self.openVideoBtn = Button(self.videoMenu, text="Open Video", font=("Helvetica", 15), height=1, width=10, bg="silver", command=self.play_video, state="disabled")
        self.openVideoBtn.grid(row=0, column=0)

        self.playVideoBtn = Button(self.videoMenu, text="Play", font=("Helvetica", 15), height=1, width=10, bg="silver", command=self.continue_video, state="disabled")
        self.playVideoBtn.grid(row=0, column=1)

        self.pauseVideoBtn = Button(self.videoMenu, text="Pause", font=("Helvetica", 15), height=1, width=10, bg="silver", command=self.stop_video, state="disabled")
        self.pauseVideoBtn.grid(row=0, column=2)

    def createVideoArea(self):
        
        self.videoArea = Canvas(self.appFrame, width=600, height=600)
        self.videoArea.pack()

    def updateVideoArea(self):
        self.videoArea["width"] = self.video_capture.get(3)
        self.videoArea["height"] = self.video_capture.get(4)

    def createMenu(self):

        self.setTimeFrame = Frame(self.appFrame)
        self.setTimeFrame.pack()
        
        
        # self.hourLabel = Label(self.setTimeFrame, font =("Helvetica", 15), text = "Hour")
        # self.hourLabel.grid(row=0, column=0)

        # self.setHour = Spinbox(self.setTimeFrame, from_=0, to=24)
        # self.setHour.grid(row=1, column=0)

        # Instead of time I added frame and dist
        self.minuteLabel = Label(self.setTimeFrame, font =("Helvetica", 15), text = "Distance")
        self.minuteLabel.grid(row=0, column=0)

        self.setMinute = Spinbox(self.setTimeFrame, from_=0, to=300)
        self.setMinute.grid(row=1, column=0)

        self.secondLabel = Label(self.setTimeFrame, font =("Helvetica", 15), text = "Frames")
        self.secondLabel.grid(row=0, column=2)

        self.setSecond = Spinbox(self.setTimeFrame, from_=0, to=300)
        self.setSecond.grid(row=1, column=2)

        self.setTimeBtn = Button(self.appFrame, text="Set Threshold", font=("Helvetica", 15), height=1, width=10, bg="silver", command=lambda: [self.set_time(), self.enable_buttons()])
        self.setTimeBtn.pack()

    def set_time(self):
        # self.hour = int(self.setHour.get())
        self.distance = int(self.setMinute.get())
        self.frame = int(self.setSecond.get())

        print(f"Setting distance threshold to {self.distance} and frame threshold to {self.frame}.")

    def enable_buttons(self):
        if self.distance and self.frame:
            state = "normal"
        else:
            state = "disabled"

        self.openVideoBtn["state"] = state
        self.playVideoBtn["state"] = state
        self.pauseVideoBtn["state"] = state
    
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
        self.updateVideoArea()
        fps_start_time = datetime.datetime.now()
        fps = 0
        total_frames = 0
        # tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)
        tracker = CentroidTracker(maxDisappeared=5, maxDistance=90)
        stationary_times = {}
        previous_position = {}
        prev_box = []
        prev_conf = []


        self.playing = True
        if self.video_capture:
            self.playing = True
            while self.playing:
                ret, frame = self.video_capture.read()
                if not ret:
                    break
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
                dangerous_text = "Dangerous Objects: {}".format(len([obj for obj in stationary_times if stationary_times[obj] > 10]))

                cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                cv2.putText(frame, tfps_text, (5, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                cv2.putText(frame, dangerous_text, (5, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

                bboxes = []
                for i in indices:
                    bboxes.append(boxes[i])
                objects = tracker.update(bboxes)
                for (objectId, bbox) in objects.items():
                    x1, y1, x2, y2 = bbox
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)
                    if objectId in previous_position:
                        prevx1, prevy1, prevx2, prevy2 = previous_position[objectId]
                        prev_centerx = (prevx1 + prevx2) / 2
                        prev_centery = (prevy1 + prevy2) / 2
                        cur_centerx = (x1 + x2) / 2
                        cur_centery = (y1 + y2) / 2
                        distance = np.sqrt((prev_centerx - cur_centerx)**2 + (prev_centery - cur_centery)**2)
                        # This is the same code, just using a user input threshold
                        if distance < self.distance:
                            if objectId in stationary_times:
                                stationary_times[objectId] += 1
                            else:
                                stationary_times[objectId] = 1
                    else:
                        previous_position[objectId] = (x1, y1, x2, y2)
                        stationary_times[objectId] = 1

                    if objectId in stationary_times:
                        # This is the same code, just using a user input threshold
                        if stationary_times[objectId] > self.frame:
                            print("ALERT")
                            print(objectId)
                            print("STATIONARY")
                            print(stationary_times[objectId])
                            cv2.rectangle(frame, (x1,y1), (x1+x2, y1+y2), (0,0,255), 4)
                            text = "DANGER: {}".format(objectId)
                            cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
                        else:
                            print("NOT STATIONARY")
                            print(stationary_times[objectId])
                            cv2.rectangle(frame, (x1,y1), (x1+x2, y1+y2), (0,255,0), 2)
                            text = "ID: {}".format(objectId)
                            cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
                   

                self.display_frame(frame)
                self.appFrame.update_idletasks()
                self.appFrame.update()
            self.video_capture.release()

    def stop_video(self):
        if not self.playing:
            return
        
        self.playing = False

    def continue_video(self):
        if self.playing:
            return
        
        self.playing = True


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