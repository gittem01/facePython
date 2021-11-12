from helpers import *

vs = WebcamVideoStream(0)
vs.start()

# default 0.6
# lower value means more strict face recognition
tolerance = 0.5

faceMap = loadDatas()
faceNames = faceMap.keys()

while not vs.stopped:
    print("-" * 30)
    t = time.time()
    boxes = face_recognition.face_locations(vs.frame, model="hog")
    encodings = face_recognition.face_encodings(vs.frame, boxes)
    
    print(f"Face recognition time: {time.time() - t}")
    
    j = 0
    t = time.time()
    for e in encodings:
        for name in faceNames:
            if np.sum(face_recognition.compare_faces(faceMap[name], e, tolerance)) > dropTo * 0.5:
                print(f"Current face name: {name}")
                break
        j += 1
    if j == 0:
        print("No face found")
    print(f"Face cross checking time: {time.time() - t}")
    
cv2.destroyAllWindows()
