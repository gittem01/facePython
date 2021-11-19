from helpers import *

vs = Capture(0)

fd = FaceDetector(vs)
fd.start()

while not vs.stopped:
    vs.update()
    
cv2.destroyAllWindows()
