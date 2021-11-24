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
    for e in encoding[:-1]:
        f.write(str(e) + " ")
    f.write(str(encoding[-1]) + "\n")

f.close()