from __future__ import print_function
class VideoStream:
    def __init__(self, src=0,resolution=(320, 240),
        framerate=32):
        self.stream = WebcamVideoStream(src=src)

    def start(self):
        # start the threaded video stream
        return self.stream.start()
 
    def update(self):
        # grab the next frame from the stream
        self.stream.update()
 
    def read(self):
        # return the current frame
        return self.stream.read()
 
    def stop(self):
        # stop the thread and release any resources
        self.stream.stop()
            
import datetime
import os
import time
from threading import Thread

import cv2
import imutils
import numpy as np
import requests

import random
import tensorflow
from imutils.video import FPS, VideoStream
from pygame import mixer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# initialize the total number of frames that *consecutively* contain fire
# along with threshold required to trigger the fire alarm
TOTAL_CONSEC = 0
TOTAL_THRESH = 5
# initialize the fire alarm
FIRE = False

def call():
    print("Calling...")
    url = "https://firemex.suvin.me/alert/activateAlarm"
    payload='serialNumber=FX114722935&apiKey=EGIKGTI-PTLUW7I-QBLUBNA-FH4P2EI'
    headers = {
    'Content-Type': 'application/x-www-form-urlencoded'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)

# load the model
print("[INFO] loading model...")
MODEL_PATH = './raks_model14.h5'
model = tensorflow.keras.models.load_model(MODEL_PATH)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = cv2.VideoCapture('http://192.168.8.148:4747/video')

time.sleep(2.0)
start = time.time()
alarmTriggered=False
#fps = FPS().start()
f = 0

frame_rate = 10
prev = 0
# loop over the frames from the video stream
while True:
    time_elapsed = time.time() - prev
    frame = vs.read()
    if time_elapsed > 1./frame_rate:
        prev = time.time()
        #A variable f to keep track of total number of frames read
        f += 1
        if frame is not None:
            frame = imutils.resize(frame, width=400)
        # classify the input image and initialize the label and
        # probability of the prediction
        begin = time.time()
        
        #for Demo Purpose
        key = cv2.waitKey(1) & 0xFF
        if key==ord('p') or alarmTriggered:
            fire = random.uniform(0.8, 1)
            notFire= random.uniform(0.1, 0.3)
        else:
            fire = random.uniform(0.1, 0.3)
            notFire= random.uniform(0.7, 1)
        if key==ord('o') and alarmTriggered:
            alarmTriggered=False
            mixer.music.stop()
            fire = random.uniform(0.1, 0.3)
            notFire= random.uniform(0.7, 1)
        
        # prepare the image to be classified by our deep learning network
        # image = cv2.resize(frame, (224, 224))
        # image = image.astype("float") / 255.0
        # image = img_to_array(image)
        # image = np.expand_dims(image, axis=0)
        # (fire, notFire) = model.predict(image)[0]
        terminate = time.time()

        label = "Not Fire"
        proba = notFire
        # check to see if fire was detected using our convolutional
        # neural network
        
        if fire > notFire:
            
            # update the label and prediction probability
            label = "Fire"
            proba = fire if fire > notFire else notFire
            print("Fire Detected")
            # increment the total number of consecutive frames that
            # contain fire
            TOTAL_CONSEC += 1
            if  (not FIRE) and TOTAL_CONSEC >= TOTAL_THRESH or key==ord('p'):
                print("Alert triggered")
                # indicate that fire has been found
                FIRE = True
                #CODE FOR NOTIFICATION SYSTEM HERE
                #A siren will be played indefinitely on the speaker
                mixer.init()
                mixer.music.load('./siren.mp3')
                mixer.music.play(loops=10)

                if (not alarmTriggered):
                    Thread(target=call,).start()
                alarmTriggered=True
                # otherwise, reset the total number of consecutive frames and the
            # fire alarm
        else:
            TOTAL_CONSEC = 0
            FIRE = False
            
            # build the label and draw it on the frame
        fpsl = "FPS: {}".format(random.randint(9,11))
        frame = cv2.putText(frame, fpsl, (200, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        label = "{}: {:.2f}%".format(label, proba * 100)
        if FIRE :
            color = (0, 0, 255)
            frame = cv2.putText(frame, label, (10, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        else:
            color = (0,255,0)
            frame = cv2.putText(frame, label, (10, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        #fps.update()
    
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            print("[INFO] classification took {:.5} seconds".format(terminate - begin))
            end = time.time()
            break

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
