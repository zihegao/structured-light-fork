from GrayCodesWindow import getImageIteration, destroyW
from CaptureImage import SaveImage, capture_and_save_image
import os
import numpy as np
from subprocess import Popen
import cv2
from kinectImageClass import KinectImageClass
import time
kic = KinectImageClass("C:/Program Files/OpenNI2/Samples/Bin")
import structuredlight as sl

DETACHED_PROCESS = 0x00000008 
BaseOutputDirBeforeNew = "./captures/"
SubCaptDir = "c_"
SaveFormat = ".jpg"
WINDOW_NAME="phaseshift"


def imShowAndCapture(cap, img_pattern, delay=250):
    cv2.imshow(WINDOW_NAME, img_pattern)
    cv2.waitKey(delay)
    time.sleep(7)
    ret, img_frame = cap.read()
    img_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)    
    return img_gray


#Capture Path: testcap
GrayCodeConverterPath = "../DecodeGrayImages/DecodeGrayImages"
cFolder = input("Enter capture folder: ") 
BaseOutputDir = BaseOutputDirBeforeNew +cFolder+"/"
width = 1920
height = 1080

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

PhaseShift = True

while DoNextIteration:
    currentI += 1
    DoNextIteration = False
    #CamDirOut = BaseOutputDir+SubCaptDir+str(currentI)+"/"
    CamDirOut = BaseOutputDir
    
    if PhaseShift is False:
        for imgnr in getImageIteration(FirstIteration):
            if imgnr == "w":
                kic.capture_image(CamDirOut+"kinect_")
            #SaveImage(CamDirOut+imgnr+SaveFormat)
            capture_and_save_image(CamDirOut+imgnr+SaveFormat)

            DoNextIteration=True

    if PhaseShift is True:
        cap = cv2.VideoCapture(0)
        phaseshifting = sl.PhaseShifting(num=3)
        # Generate and Decode x-coord
        # Generate
        imlist_posi_pat = phaseshifting.generate((width, height))
        imlist_nega_pat = sl.invert(imlist_posi_pat)

        imgToDisplay = np.zeros((768, 1024), dtype = np.uint8)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.imshow(WINDOW_NAME, imgToDisplay)
        cv2.waitKey(0)

        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(WINDOW_NAME, imgToDisplay)

        # Capture
        imlist_posi_pat = [imShowAndCapture(cap, img) for img in imlist_posi_pat]
        i=0
        for img in imlist_posi_pat:
            cv2.imwrite("w_%d" % (i) + SaveFormat, img)
            i+= 1


    if DoNextIteration:
        Popen(["python3 ConvertRawImage.py "+CamDirOut +" "+ SaveFormat +" "+ GrayCodeConverterPath], shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)
    FirstIteration=False
destroyW()
kic.unload()

