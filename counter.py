import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

cv2.namedWindow("preview")
capture = cv2.VideoCapture("sample_short.mp4")


if capture.isOpened(): # try to get the first frame
   ok, frame = capture.read()
else:
   ok = False

while True :
   ok, frame = capture.read()
   img_blur = cv2.medianBlur(frame,15)
   img_gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
   img_cirlces = cv2.cvtColor(img_gray,cv2.COLOR_GRAY2BGR)
   circles =   cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT,1,120,param1=100,param2=30,minRadius=20,maxRadius=200)

   if circles is None:
      cv2.imshow("preview", frame)
      continue

   #circles = np.uint16(np.around(circles))

   for i in circles[0,:]:
      print(i)
      cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),1) # draw the outer circle
      cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3) # draw the center of the circle


   cv2.imshow("preview", frame)

   if cv2.waitKey(1) & 0xFF == ord('q'):
      break

capture.release()


