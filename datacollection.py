src/datacollection.py
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
offset = 50
imgSize = 500
counter = 0

folder = "/Users/VICTUS/Desktop/Sign language/Data/Okay"

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # Initialize imgWhite

    if hands:
        imgCombined = np.zeros_like(img)  # Initialize imgCombined to combine both hands
        for hand in hands:  # Iterate over each detected hand
            x, y, w, h = hand['bbox']

            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            imgCombined[y - offset:y + h + offset, x - offset:x + w + offset] = imgCrop

        # Resize and paste the combined image onto imgWhite
        imgCombined = cv2.resize(imgCombined, (imgSize, imgSize))
        imgWhite[:imgCombined.shape[0], :imgCombined.shape[1]] = imgCombined

        cv2.imshow('ImageCombined', imgCombined)
        cv2.imshow('ImageWhite', imgWhite)


    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
