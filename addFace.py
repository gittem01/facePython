from helpers import *

if not os.path.exists("data"):
    os.mkdir("data")

vs = WebcamVideoStream(0)
vs.start()

encodings = []
capturedImages = []

# Randomly selects 20 face data out of all data
n = 1

while not vs.stopped:
    capturedFrame = vs.frame.copy()
    
    boxes = face_recognition.face_locations(capturedFrame, model="hog")
    encoding = face_recognition.face_encodings(capturedFrame, boxes)
    if len(encoding) == 1:
        encodings.append(encoding[0])
        capturedImages.append(capturedFrame)
        print(f"Number of captured faces: {n}", end="\r")
        n += 1

print()    

cv2.destroyAllWindows()

inp = input("Enter a name for the face: ")

f = open("data/" + inp + ".data", "w")

random.shuffle(encodings)
encodings = encodings[:dropTo]

for encoding in encodings:
    i = 0
    for e in encoding:
        i += 1
        if i != 128:
            f.write(str(e) + " ")
        else:
            f.write(str(e) + "\n")

f.close()