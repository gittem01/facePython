import face_recognition
import cv2
import numpy as np
import time
import random
import os
from threading import Thread

dropTo = 10

class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        
    def start(self):
        Thread(target=self.update, args=()).start()

    def update(self):
        while True:
            self.grabbed, self.frame = self.stream.read()
            self.frame = cv2.resize(self.frame, (480, 360), interpolation = cv2.INTER_AREA)
            
            cv2.imshow("Video", self.frame)
            key = cv2.waitKey(1)
            if (key == ord("q")):
                break
        
        self.stopped = True


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