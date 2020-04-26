import cv2
import numpy as np

cap = cv2.VideoCapture("/home/priya/Documents/pj2/fire_videos.1406/pos/posVideo6.875.avi")
#cap = cv2.VideoCapture("/home/priya/Documents/pj2/fire_videos.1406/pos/posVideo6.875.avi")
#/home/priya/Documents/pj2/fire_videos.1406/pos
while(1):

    # Take each frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([0,100,100])
    upper_blue = np.array([10,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()