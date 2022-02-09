# facePython

Project uses python face_recognition library to classify faces.
OpenCV is used for displaying the camera input.
OpenCV display images by using main thread and face recognition part (computationally intensive)
is handled in a seperate thread, thanks to that FPS does not suffer.

## Usage
 - Run addFace.py file in order to add a new list
 - Run  main.py to classify faces
