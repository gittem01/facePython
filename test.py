import face_recognition
import cv2
import numpy as np
import time
from threading import Thread

OS = "pc" ## pc or pi

if OS == "pi":
    from picamera.array import PiRGBArray
    from picamera import PiCamera

class WebcamVideoStream:
    def __init__(self, src=0):
        if OS == "pi":
            self.camera = PiCamera()
            self.camera.brightness = 60
            self.camera.resolution = (640, 480)
            self.camera.framerate = 32
            self.rawCapture = PiRGBArray(self.camera, size=(640, 480))
            time.sleep(0.1)
            self.frame = np.empty((480, 640, 3), dtype=np.uint8)
        elif OS == "pc":
            self.stream = cv2.VideoCapture(src)
            self.grabbed, self.frame = self.stream.read()
        
        self.stopped = False
        
    def start(self):
        Thread(target=self.update, args=()).start()

    def update(self):
        while True:
            if OS == "pi":
                self.camera.capture(self.frame, format="bgr")
            elif OS == "pc":
                self.grabbed, self.frame = self.stream.read()
                #self.frame = cv2.resize(self.frame, (480, 360), interpolation = cv2.INTER_AREA)
            
            cv2.imshow("Frame", self.frame)
            key = cv2.waitKey(1)
            if (key == ord("q")):
                break
        
        self.stopped = True

dataFile = open("data.txt", "r")

dataArray = dataFile.readlines()

arr = np.empty((1, 128))

b = 1
for s in dataArray:
    s = s[:-1]
    sSplit = s.split(" ")
    for d in sSplit:
        if len(d) > 3:
            arr[0][b-1] = float(d)
            b += 1

vs = WebcamVideoStream(0)
vs.start()

# default 0.6
# lower value means more strict face recognition
tolerance = 0.5

i = 1
while not vs.stopped:
    t = time.time()
    boxes = face_recognition.face_locations(vs.frame, model="hog")
    encodings = face_recognition.face_encodings(vs.frame, boxes)
    
    print(f"Face recognition time: {time.time() - t}")
    
    i += 1
    j = 0
    t = time.time()
    for e in encodings:
        print(face_recognition.compare_faces(arr, e, tolerance))
        j += 1
        print(f"Face found {j}")
    if j == 0:
        print("No face found")
    print(f"Face cross checking time: {time.time() - t}")
    
cv2.destroyAllWindows()