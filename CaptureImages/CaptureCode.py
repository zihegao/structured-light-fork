from GrayCodesWindow import getImageIteration, destroyW
from CaptureImage import SaveImage 
import os
from subprocess import Popen
import cv2
from kinectImageClass import KinectImageClass
kic = KinectImageClass("C:/Program Files/OpenNI2/Samples/Bin")

def capture_and_save_image(file_name):
    # Initialize the camera
    cap = cv2.VideoCapture(1)  # 0 for the default camera, you can change it if needed

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


DETACHED_PROCESS = 0x00000008 
BaseOutputDirBeforeNew = "../captures/"
SubCaptDir = "c_"
SaveFormat = ".jpg"

#Capture Path: testcap
GrayCodeConverterPath = "../DecodeGrayImages/DecodeGrayImages"
cFolder = input("Enter capture folder: ") 
BaseOutputDir = BaseOutputDirBeforeNew +cFolder+"/"

currentI = -1
if os.path.isdir(BaseOutputDir):
    print("isdir")
    for folder in os.listdir(BaseOutputDir):
        print(folder)
        if os.path.isdir(BaseOutputDir+folder) and folder.startswith(SubCaptDir):
            try:
                currentI = max(currentI, int(folder[len(SubCaptDir):]))
            except:
                pass
DoNextIteration = True
FirstIteration = True
while DoNextIteration:
    currentI += 1
    DoNextIteration = False
    #CamDirOut = BaseOutputDir+SubCaptDir+str(currentI)+"/"
    CamDirOut = BaseOutputDir+"testcap/"
    for imgnr in getImageIteration(FirstIteration):
        if imgnr == "w":
            kic.capture_image(CamDirOut+"kinect_")
        #SaveImage(CamDirOut+imgnr+SaveFormat)
        capture_and_save_image(CamDirOut+imgnr+SaveFormat)

        DoNextIteration=True
    if DoNextIteration:
        Popen(["python3 ConvertRawImage.py "+CamDirOut +" "+ SaveFormat +" "+ GrayCodeConverterPath], shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)
    FirstIteration=False
destroyW()
kic.unload()

