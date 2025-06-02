# file is used to determine if a camera is properly connected and is available
# will print to consolde that 'index' camera is available if so, else none will be 

import cv2

def get_available_cameras():
    available_cameras = []
    # Check for 5 cameras 
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

cameras = get_available_cameras()
if cameras:
    print("Available Cameras:", cameras)
else:
    print("No cameras found.")
