import cv2
import pygame
from pygame.locals import *
from tkinter import Tk, filedialog
from PIL import Image
from io import BytesIO

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
        cap = cv2.VideoCapture(0)
        self.display_video(cap)

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
