import os
import cv2
import time

GPHOTO_PATH = "gphoto2"

########## Uncomment this block for a windows pc ######
os.environ["IOLIBS"] = r'/win32/iolibs'
os.environ["CAMLIBS"] = r'/win32/camlibs'

GPHOTO_PATH="C:/CameraLibs/win32/gphoto2.exe"

TEMP_GPHOTO_DIR = "capture/out.temp"
GPHOTO_PARAMS = " --capture-image-and-download --filename "


def TakeImage():
    if os.path.isfile(TEMP_GPHOTO_DIR):
        os.remove(TEMP_GPHOTO_DIR)

    os.system(GPHOTO_PATH+GPHOTO_PARAMS+TEMP_GPHOTO_DIR)

    img = cv2.imread(TEMP_GPHOTO_DIR)

    os.remove(TEMP_GPHOTO_DIR)
    
    return img


def SaveImage(FileName):
    if os.path.isfile(FileName):
        os.remove(FileName)

    return os.system(GPHOTO_PATH + GPHOTO_PARAMS + FileName)

def capture_and_save_image(file_name):
    # Initialize the camera
    cap = cv2.VideoCapture(0)  # 0 for the default camera, you can change it if needed

    time.sleep(0)  # Pause execution for one second

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Couldn't open camera.")
        return

    # Capture a frame
    ret, frame = cap.read()

    # Check if the frame is captured successfully
    if not ret:
        print("Error: Couldn't capture frame.")
        cap.release()
        return

    # Save the captured frame as an image
    cv2.imwrite(file_name, frame)

    # Release the camera
    cap.release()

    print(f"Image captured and saved as {file_name}.")

# Example usage:
# capture_and_save_image("captured_image.jpg")


