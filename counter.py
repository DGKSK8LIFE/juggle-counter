import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Boy in sample.mp4 does 158 juggles

cv2.namedWindow("preview")
capture = cv2.VideoCapture("sample_medium.mp4")

circle_y = []

if capture.isOpened(): # try to get the first frame
   ok, frame = capture.read()
else:
   ok = False

num_frames = 0
prev_y, prev_rad = 0, 0

while True :
      ok, frame = capture.read()

      # When the video ends, the progam exits without an error
      if ok == False:
         break
      img_blur = cv2.medianBlur(frame,15)
      img_gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
      img_cirlces = cv2.cvtColor(img_gray,cv2.COLOR_GRAY2BGR)
      circles =   cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=225, param1=100, param2=30, minRadius=20, maxRadius=150)
      
      if circles is None:
         cv2.imshow("preview", frame)
         continue

      circles = np.uint64(np.around(circles))

      for i in circles[0,:]:
         #if(abs(int(i[2]) - prev_rad)) < 15:
         print(abs(int(i[1]) - prev_y))
         circle_y.append(i[1])
         cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),1) # draw the outer circle
         cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3) # draw the center of the circle

         prev_y = i[1]



      cv2.imshow("preview", frame)
      num_frames += 1

      if cv2.waitKey(1) & 0xFF == ord('q'):
         break

capture.release()

sample = np.array(circle_y)

peaks, _ = find_peaks(sample) # Returns the indexes of the peaks

print(peaks)
print(len(peaks))
print(num_frames)
print(len(circles))

plt.plot(sample)
# plt.plot(range(num_frames), sample)
plt.plot(peaks, sample[peaks], "x")
plt.show()

#TODO: Remove noise/circle detection mistakes
#TODO: Add threshold so that small unessesary ball fluctuations are not counted as peaks

