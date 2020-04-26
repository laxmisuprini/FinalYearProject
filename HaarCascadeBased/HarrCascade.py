import numpy as np
import cv2
import serial
import time


fire_cascade = cv2.CascadeClassifier('/home/priya/Documents/pj2/fire_hcc/data/cascade.xml')
#fire_detection.xml file & this code should be in the same folder while running the code


#cap = cv2.VideoCapture("/home/priya/Downloads/Pexels Videos 1777623.mp4")
#cap = cv2.VideoCapture("/home/priya/Documents/pj2/fire_videos.1406/pos/posVideo6.875.avi")
cap = cv2.VideoCapture("/home/priya/Documents/pj2/fire_videos.1406/pos/posVideo6.875.avi")
cap.set(3,640) # set Width
cap.set(4,480) # set Height
while True:
    #ser1.write('0')
    ret, img = cap.read()
    if not ret:
        print("video not captured")
        break
    #cv2.imshow('imgorignal',img)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    fire = fire_cascade.detectMultiScale(img, 1.2, 5)
    print(fire)
    for (x,y,w,h) in fire:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        print ('Fire is detected..!')
        #ser1.write('p')
        time.sleep(0.2)
        
    cv2.imshow('video',img)
    #ser1.write('s')
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
