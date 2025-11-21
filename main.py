src/main.py
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import tkinter as tk
from googletrans import Translator

class HandGestureGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Hand Gesture Chat")
        self.master.configure(bg="#1E1E1E")

        self.text_box = tk.Text(master, state=tk.DISABLED, wrap=tk.WORD, bg="#1E1E1E",
                                fg="#FFFFFF", font=("Arial", 14, "bold"))
        self.text_box.pack(pady=10)

        self.translator = Translator()

        self.cap = cv2.VideoCapture(0)
        self.detector = HandDetector(maxHands=1)
        self.classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

        self.offset = 20
        self.imgSize = 300

        self.labels = ["A", "B", "C", "Hi!", "My", "Name is", "My name is MIRSHAN TM",
                       "Thank you", "Y", "Boy", "Girl", "Yes"]

        self.min_interval = 5
        self.last_prediction_time = time.time() - self.min_interval
        self.start_detection = False
        self.delay_before_detection = 3

        self.detected_labels = []
        self.delete_on_space = False

        self.update_gui()
        master.bind('<space>', self.delete_label)
        master.bind('<BackSpace>', self.delete_last_word)

    def update_gui(self):
        success, img = self.cap.read()
        img_output = img.copy()
        hands, img = self.detector.findHands(img)

        if not self.start_detection:
            cv2.putText(img, f"Waiting {self.delay_before_detection} seconds", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            current_time = time.time()

            if current_time - self.last_prediction_time >= self.delay_before_detection:
                self.start_detection = True
                self.last_prediction_time = current_time

        elif hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            y1 = max(y - self.offset, 0)
            y2 = min(y + h + self.offset, img.shape[0])
            x1 = max(x - self.offset, 0)
            x2 = min(x + w + self.offset, img.shape[1])

            img_crop = img[y1:y2, x1:x2]

            img_white = np.ones((self.imgSize, self.imgSize, 3), np.uint8) * 255
            img_crop_shape = img_crop.shape

            aspect_ratio = h / w

            if aspect_ratio > 1:
                k = self.imgSize / h
                w_cal = math.ceil(k * w)
                img_resize = cv2.resize(img_crop, (w_cal, self.imgSize))
                w_gap = math.ceil((self.imgSize - w_cal) / 2)
                img_white[:, w_gap: w_cal + w_gap] = img_resize

            else:
                k = self.imgSize / w
                h_cal = math.ceil(k * h)
                img_resize = cv2.resize(img_crop, (self.imgSize, h_cal))
                h_gap = math.ceil((self.imgSize - h_cal) / 2)
                img_white[h_gap: h_cal + h_gap, :] = img_resize

            current_time = time.time()

            if current_time - self.last_prediction_time >= self.min_interval:
                prediction, index = self.classifier.getPrediction(img_white, draw=False)

                label_name = self.labels[index]
                translation = self.translator.translate(label_name, dest='en').text

                self.text_box.config(state=tk.NORMAL)
                self.text_box.insert(tk.END, translation + " ")
                self.text_box.config(state=tk.DISABLED)

                self.detected_labels.append(label_name)
                self.last_prediction_time = current_time

        cv2.imshow('Image', img_output)
        self.master.after(1, self.update_gui)

    def delete_label(self, event):
        if self.detected_labels:
            self.detected_labels.pop()
            updated_text = ' '.join(self.detected_labels)

            self.text_box.config(state=tk.NORMAL)
            self.text_box.delete(1.0, tk.END)
            self.text_box.insert(tk.END, updated_text + " ")
            self.text_box.config(state=tk.DISABLED)

    def delete_last_word(self, event):
        if self.delete_on_space:
            if self.detected_labels:
                self.detected_labels.pop()
                updated_text = ' '.join(self.detected_labels)

                self.text_box.config(state=tk.NORMAL)
                self.text_box.delete(1.0, tk.END)
                self.text_box.insert(tk.END, updated_text + " ")
                self.text_box.config(state=tk.DISABLED)

            self.delete_on_space = False

if __name__ == "__main__":
    root = tk.Tk()
    gui = HandGestureGUI(root)
    root.mainloop()
