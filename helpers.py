import face_recognition
import cv2
import numpy as np
import time
import random
import os
from threading import Thread

# drops number of captured face data to \(dropTo)
# by randomly selecting \(dropTo) datas from the data list
dropTo = 1

class FaceDetector:
    def __init__(self, captureClass):
        self.captureClass = captureClass
        
        # default 0.6
        # lower value means more strict face recognition
        self.tolerance = 0.5
        
        # for facedata addition
        self.encodings = []
        self.capturedImages = []

        self.done = False

    def start(self):
        Thread(target=self.update, args=()).start()

    def startFaceAddition(self):
        Thread(target=self.addFaceUpdate, args=()).start()

    def addFaceUpdate(self):
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
        faceMap = loadDatas()
        faceNames = faceMap.keys()

        while not self.captureClass.stopped:
            if (type(self.captureClass.frame) == type(None)):
                continue
            print("-" * 30)
            t = time.time()
            boxes = face_recognition.face_locations(self.captureClass.frame, model="hog")
            encodings = face_recognition.face_encodings(self.captureClass.frame, boxes)
            
            print(f"Face recognition time: {time.time() - t}")
            
            if len(encodings) == 0:
                print("No face found")
            else:
                print(f"{len(encodings)} face found")

            t = time.time()
            for e in encodings:
                for name in faceNames:
                    if np.sum(face_recognition.compare_faces(faceMap[name], e, self.tolerance)) > dropTo * 0.5:
                        print(f"Detected faces: {name}")
                        break

            print(f"Face cross checking time: {time.time() - t}")

        self.done = True

class Capture:
    def __init__(self, src=0, scale=0.5):
        self.stream = cv2.VideoCapture(src)
        self.scale = scale
        self.frame = None
        self.stopped = False

    def update(self):
        key = cv2.waitKey(1)
        if key == ord("q") or key == 27: # 27 : ESC key
            self.stopped = True
            cv2.destroyAllWindows()
            return

        _, self.frame = self.stream.read()

        if (self.scale != 1):
            self.frame = cv2.resize(self.frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

        cv2.imshow("Video", self.frame)


def loadSingleData(dataFile):
    f = open(dataFile, "r")
    rl = f.readlines()

    arr = np.empty((len(rl), 128))

    i = 0
    for data in rl:
        data = data[:-1]
        d = data.split(" ")
        j = 0
        for val in d:
            arr[i][j] = float(val)
            j += 1
        i += 1
    return arr

def loadDatas():
    if not os.path.exists("data"):
        return {}

    files = os.listdir("data")
    
    returnMap = {}
    for file in files:
        if file.split(".")[-1] != "data":
            continue
        data = loadSingleData("data/" + file)
        fileName = "".join(file.split(".")[:-1])
        returnMap[fileName] = data
    return returnMap