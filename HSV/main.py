import cv2
import numpy as np 
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


capture = cv2.VideoCapture("../Resources/sample_medium.mp4")

capture.set(3, 640)
capture.set(4, 480)

num_frames = 0

while True:
    ok, frame = capture.read()

    if ok == False:
        break

    if num_frames == 0:
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    cv2.imshow("preview", frame)

    num_frames += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break