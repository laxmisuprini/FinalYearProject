import cv2
import imutils
import keras
from keras.preprocessing.image import img_to_array
from keras.models import load_model
#from imutils.video import VideoStream
#from imutils.video import FPS
import math
from threading import Thread
import numpy as np
import time
import os
from playsound import playsound
#from picamera.array import PiRGBArray
#from picamera import PiCamera
#from imutils.video.pivideostream import PiVideoStream
import datetime
#from pygame import mixer
 

# initialize the total number of frames that *consecutively* contain fire
# along with threshold required to trigger the fire alarm
TOTAL_CONSEC = 0
TOTAL_THRESH = 20
# initialize the fire alarm
FIRE = False


# load the model
print("[INFO] loading model...")
MODEL_PATH = '/home/priya/Documents/Final_year project/Inferno-Realtime-Fire-detection-using-CNNs-master/saved_model/raks_model14.h5'
model = keras.models.load_model(MODEL_PATH)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
vs =cv2.VideoCapture("/home/priya/Documents/pj2/fire_videos.1406/pos/posVideo4.873.avi")
vs.set(3,640) # set Width
vs.set(4,480)
time.sleep(2.0)
start = time.time()
#fps = FPS().start()
f = 0
width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH));
height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = vs.get(cv2.CAP_PROP_FPS)
frame_time = round(1000/fps);
windowName = "fire detection"
keepProcessing = True;
# loop over the frames from the video stream
while keepProcessing:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    start_t = cv2.getTickCount();

    ret,frame = vs.read()
    if not ret:
        print("no video")
        break;
    #A variable f to keep track of total number of frames read
    f += 1
    #frame = imutils.resize(frame, width=400)
    #frame = np.array(array).astype(np.uint8)
    # prepare the image to be classified by our deep learning network
    dim=(224,224)
    image = cv2.resize(frame,dim , interpolation = cv2.INTER_AREA)
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
 
    # classify the input image and initialize the label and
    # probability of the prediction
    begin = time.time()
    (fire, notFire) = model.predict(image)[0]
    terminate = time.time()

    label = "Not Fire"
    proba = notFire
    # check to see if fire was detected using our convolutional
    # neural network
    if fire > notFire:
        # update the label and prediction probability
        label = "Fire"
        proba = fire
 
        # increment the total number of consecutive frames that
        # contain fire
        TOTAL_CONSEC += 1
        if not FIRE and TOTAL_CONSEC >= TOTAL_THRESH:
            # indicate that fire has been found
            FIRE = True
            print("fire!!!!")
            #playsound("/home/priya/Downloads/siren.mp3")
            #CODE FOR NOTIFICATION SYSTEM HERE
	    #A siren will be played indefinitely on the speaker
            '''mixer.init()
            mixer.music.load('/home/pi/Desktop/siren.mp3')
            mixer.music.play(-1)
            '''
            # otherwise, reset the total number of consecutive frames and the
    # fire alarm
    else:
        TOTAL_CONSEC = 0
        FIRE = False
        
        # build the label and draw it on the frame
    label = "{}: {:.2f}%".format(label, proba * 100)
    frame = cv2.putText(frame, label, (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    #fps.update()
 
    # if the `q` key was pressed, break from the loop
    stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000;
    key = cv2.waitKey(max(2, frame_time - int(math.ceil(stop_t)))) & 0xFF;
    if (key == ord('x')):
        print("[INFO] classification took {:.5} seconds".format(terminate - begin))
        end = time.time()
        keepProcessing = False;
    elif (key == ord('f')):
        cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);

# do a bit of cleanup
print("[INFO] cleaning up...")
seconds = end - start
print("Time taken : {0} seconds".format(seconds))
fps  = f/ seconds
print("Estimated frames per second : {0}".format(fps))
#fps.stop()
#print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
#print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()