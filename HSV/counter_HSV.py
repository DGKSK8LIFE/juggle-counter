import cv2
import numpy as np 
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

capture = cv2.VideoCapture("../Resources/1500.mp4")

capture.set(3, 640)
capture.set(4, 480)

def empty(a):
    pass

def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


while True:
    ok, first_frame = capture.read()

    print("Press 'p' to pause the video to pick out the ball (please try to do this as soon as you clearly see the ball!)")

    if ok == False:
        break

    cv2.imshow("preview", first_frame)

    if cv2.waitKey(50) & 0xFF == ord('p'):
        break

cv2.destroyAllWindows()

print("Expirement trackbars to get the ball white and everything black. Press 'n' for next step.")

cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 640, 240)

cv2.createTrackbar("Hue min", "Trackbars", 0, 179, empty)
cv2.createTrackbar("Hue max", "Trackbars", 179, 179, empty)

cv2.createTrackbar("Saturation min", "Trackbars", 0, 255, empty)
cv2.createTrackbar("Saturation max", "Trackbars", 255, 255, empty)

cv2.createTrackbar("Value min", "Trackbars", 0, 255, empty)
cv2.createTrackbar("Value max", "Trackbars", 255, 255, empty)

while True:
    blurred_img = cv2.GaussianBlur(first_frame, (11, 11), 0)

    hsv_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)

    hue_min = cv2.getTrackbarPos("Hue min", "Trackbars")
    hue_max = cv2.getTrackbarPos("Hue max", "Trackbars")

    sat_min = cv2.getTrackbarPos("Saturation min", "Trackbars")
    sat_max = cv2.getTrackbarPos("Saturation max", "Trackbars")

    val_min = cv2.getTrackbarPos("Value min", "Trackbars")
    val_max = cv2.getTrackbarPos("Value max", "Trackbars")

    lower = np.array([hue_min, sat_min, val_min])
    upper = np.array([hue_max, sat_max, val_max])

    mask = cv2.inRange(hsv_img, lower, upper)

    bitwise_img = cv2.bitwise_and(first_frame, first_frame, mask=mask)

    cv2.imshow("Out", stackImages(0.5, ([first_frame, bitwise_img], [hsv_img, mask])))

    if cv2.waitKey(1) & 0xFF == ord('n'):
        break

cv2.destroyAllWindows()

circle_y = []

num_frames = 0

while True:
    success, frame = capture.read()

    if success == False:
        break

    blurred_frame = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    mask_frame = cv2.inRange(hsv_frame, lower, upper)
    mask_frame = mask = cv2.erode(mask_frame, None, iterations=2)
    mask_frame = cv2.dilate(mask_frame, None, iterations=2)

    contours, hierarchy = cv2.findContours(mask_frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # If we have at least one contour, look through each one and pick the biggest
    if len(contours)>0:
        largest = 0
        area = 0
        for i in range(len(contours)):
            # get the area of the ith contour
            temp_area = cv2.contourArea(contours[i])
            # if it is the biggest we have seen, keep it
            if temp_area > area:
                area = temp_area
                largest = i
        # Compute the coordinates of the center of the largest contour
        coordinates = cv2.moments(contours[largest])
        target_x = int(coordinates['m10']/coordinates['m00'])
        target_y = int(coordinates['m01']/coordinates['m00'])
        circle_y.append(target_y)
        # Pick a suitable diameter for our target (grows with the contour)
        diam = int(np.sqrt(area)/4)
        # draw on a target
        cv2.circle(frame,(target_x,target_y),diam,(0,255,0),1)
        cv2.line(frame,(target_x-2*diam,target_y),(target_x+2*diam,target_y),(0,255,0),1)
        cv2.line(frame,(target_x,target_y-2*diam),(target_x,target_y+2*diam),(0,255,0),1)

        num_frames += 1

        cv2.imshow('View', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

sample = np.array(circle_y)

peaks, _ = find_peaks(sample) # Returns the indexes of the peaks

print('Peaks:', peaks)
print('Number of juggles:', len(peaks))
print('Number of circles detected:', len(sample))
print('Num of frames:', num_frames)

plt.plot(sample)
plt.xlabel("frames")
plt.ylabel("height of ball (px)")
plt.plot(peaks, sample[peaks], "x")
plt.show()