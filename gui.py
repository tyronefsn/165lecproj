from tkinter import *
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk

def main():
    root = Tk()
    root.title("CMSC 165 Project")
    AppGUI(root)
    root.mainloop()

class AppGUI():
    def __init__(self, root):
        self.appFrame = Frame(root)
        self.appFrame.pack()

        self.video_path = None
        self.video_capture = None

        self.createVideoMenu()
        self.createVideoArea()
        self.createMenu()

    def createVideoMenu(self):
        
        self.videoMenu = Frame(self.appFrame)
        self.videoMenu.pack()

        self.openVideoBtn = Button(self.videoMenu, text="Open Video", font=("Helvetica", 15), height=1, width=6, bg="silver", command=self.openVideo)
        self.openVideoBtn.grid(row=0, column=0)

        self.playVideoBtn = Button(self.videoMenu, text="Play Video", font=("Helvetica", 15), height=1, width=6, bg="silver", command=self.play_video)
        self.playVideoBtn.grid(row=0, column=1)

        self.pauseVideoBtn = Button(self.videoMenu, text="Pause Video", font=("Helvetica", 15), height=1, width=6, bg="silver", command=self.stop_video)
        self.pauseVideoBtn.grid(row=0, column=2)



    def createVideoArea(self):
        
        self.videoArea = Canvas(self.appFrame)
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

        self.setTimeBtn = Button(self.appFrame, text="Set Time", font=("Helvetica", 15), height=1, width=5, bg="silver")
        self.setTimeBtn.pack()

    def play_video(self):
        if self.video_capture:
            while True:
                ret, frame = self.video_capture.read()
                if not ret:
                    break

                self.display_frame(frame)
                self.root.update_idletasks()
                self.root.update()

    def stop_video(self):
        if self.video_capture:
            self.video_capture.release()


    def openVideo(self):
        file_path = filedialog.askopenfilename(initialdir='../', title="Select Video File", filetypes=(("Video Files", "*.mp4"), ("all files", "*.*")))

        if file_path:
            self.video_path = file_path
            self.video_capture = cv2.VideoCapture(self.video_path)

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