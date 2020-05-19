import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Boy in sample.mp4 does 158 juggles
# sample_medium.mp4 = 24

cv2.namedWindow("preview")
capture = cv2.VideoCapture("../Resources/IMG_0134.MOV")

capture.set(3, 640)
capture.set(4, 480)

circle_y = []

if capture.isOpened(): # try to get the first frame
   ok, frame = capture.read()
else:
   ok = False

num_frames = 0
prev = [0, 0, 0]

distance = lambda x1, y1, x2, y2 : ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** .5

while True :
   ok, frame = capture.read()

   # When the video ends, the progam exits without an error
   if ok == False:
      break

   img_blur = cv2.medianBlur(frame,15)
   img_gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
   circles =   cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=225, param1=100, param2=30, minRadius=50, maxRadius=150)
   
   if circles is None:
      cv2.imshow("preview", frame)
      continue

   circles = np.uint16(np.around(circles))

   for i in circles[0,:]:
      if (abs(int(i[2]) - prev[2]) < 30) and (distance(int(i[0]), int(i[1]), prev[0], prev[1]) < 200):
         circle_y.append(i[1])
      
         cv2.circle(frame, (i[0], i[1]), i[2], (255,0,0), 3) # draw the outer circle
         cv2.circle(frame, (i[0], i[1]), 2, (0,0,255), 3) # draw the center of the circle

      prev = i

   cv2.imshow("preview", frame)
   num_frames += 1

   if cv2.waitKey(1) & 0xFF == ord('q'):
      break

capture.release()

sample = np.array(circle_y)

peaks, _ = find_peaks(sample, threshold=5, distance=1) # Returns the indexes of the peaks

print('peaks', peaks)
print('number of juggles', len(peaks))
print('number of frames', num_frames)

plt.plot(sample)
plt.plot(peaks, sample[peaks], "x")
plt.show()