import face_recognition
import cv2
import numpy as np
import time
import random
import os
from threading import Thread

dropTo = 10

class FaceDetector:
    def __init__(self, captureClass):
        self.captureClass = captureClass
        
        # default 0.6
        # lower value means more strict face recognition
        self.tolerance = 0.5

        self.faceMap = loadDatas()
        self.faceNames = self.faceMap.keys()
        
        # for facedata addition
        self.encodings = []
        self.capturedImages = []

        self.done = False

    def start(self):
        Thread(target=self.update, args=()).start()

    def startFaceAddition(self):
        Thread(target=self.addFaceUpdate, args=()).start()

    def addFaceUpdate(self):
        # Randomly selects 'n' face data out of all data
        n = 1
        while not self.captureClass.stopped:
            if (type(self.captureClass.frame) == type(None)):
                continue
            capturedFrame = self.captureClass.frame.copy()

            boxes = face_recognition.face_locations(capturedFrame, model="hog")
            encoding = face_recognition.face_encodings(capturedFrame, boxes)
            if len(encoding) == 1:
                self.encodings.append(encoding[0])
                self.capturedImages.append(capturedFrame)
                print(f"Number of captured faces: {n}", end="\r")
                n += 1
        self.done = True
        
    def update(self):
        while not self.captureClass.stopped:
            if (type(self.captureClass.frame) == type(None)):
                continue
            print("-" * 30)
            t = time.time()
            boxes = face_recognition.face_locations(self.captureClass.frame, model="hog")
            encodings = face_recognition.face_encodings(self.captureClass.frame, boxes)
            
            print(f"Face recognition time: {time.time() - t}")
            
            j = 0
            t = time.time()
            for e in encodings:
                for name in self.faceNames:
                    if np.sum(face_recognition.compare_faces(self.faceMap[name], e, self.tolerance)) > dropTo * 0.5:
                        print(f"Detected faces: {name}")
                        break

                j += 1
            if j == 0:
                print("No face found")
            print(f"Face cross checking time: {time.time() - t}")

class Capture:
    def __init__(self, src=0, scale=0.5):
        self.stream = cv2.VideoCapture(src)
        self.scale = scale
        self.frame = None
        self.stopped = False
        self.key = 0

    def update(self):
        self.key = cv2.waitKey(1)
        if self.key == ord("q") or self.key == 27:
            self.stopped = True
            cv2.destroyAllWindows()
            return
        _, self.frame = self.stream.read()
        if (self.scale != 1):
            self.frame = cv2.resize(self.frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        cv2.imshow("Video", self.frame)



def loadData(dataFile):
    f = open(dataFile, "r")
    rl = f.readlines()

    arr = np.empty((len(rl), 128))

    i = -1
    for data in rl:
        i += 1
        data = data[:-1]
        d = data.split(" ")
        j = -1
        for val in d:
            j += 1
            arr[i][j] = float(val)
    return arr

def loadDatas():
    files = os.listdir("data")

    returnMap = {}
    for file in files:
        data = loadData("data/" + file)
        fileName = "".join(file.split(".")[:-1])
        returnMap[fileName] = data
    return returnMap