from helpers import *

if not os.path.exists("data"):
    os.mkdir("data")

vs = Capture(0)

fd = FaceDetector(vs)
fd.startFaceAddition()

while not vs.stopped:
    vs.update()    

while 1:
    if not fd.done:
        continue
    else:
        break

print()
inp = input("Enter a name for the face: ")

f = open("data/" + inp + ".data", "w")

random.shuffle(fd.encodings)
encodings = fd.encodings[:dropTo]

for encoding in encodings:
    i = 0
    for e in encoding:
        i += 1
        if i != 128:
            f.write(str(e) + " ")
        else:
            f.write(str(e) + "\n")

f.close()