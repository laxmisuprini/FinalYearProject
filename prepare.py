import cv2
import os


path = "/home/priya/Documents/pj2/fire_videos.1406/pos"
if not os.path.exists('fire_cnn'):
		os.mkdir('fire_cnn')

		
vid = cv2.VideoCapture("/home/priya/Documents/pj2/fire_videos.1406/pos/pos/posVideo5.874.avi")
def getFrame(sec,vidcap):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite("fire_cnn/img"+str(count)+".jpg", image)     # save frame as JPG file
    return hasFrames
sec = 0
frameRate = 0.5 #//it will capture image in each 0.5 second
count=479
success = getFrame(sec,vid)

while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec,vid)

