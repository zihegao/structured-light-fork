# Captures undistorted Gray code images using a calibrated camera setup and saves them in structured folders.
# Automatically runs a Gray code decoder after each full capture sequence.

import os
import cv2
import numpy as np
from subprocess import Popen
from GrayImages import GrayImage
from GrayCodesWindow import getImageIteration, destroyW
from CaptureImage import SaveImage, capture_and_save_image

# load stereo calibration items
calib = np.load("../camera_calibration_out/calculated_cams_matrix.npz")
cameraMatrix = calib['cameraMatrix2']
distCoeffs = calib['distCoeffs2']
R = calib['R2']
newCameraMatrix = calib['P2']

# initialize grey code image util 
gi = GrayImage()

map1, map2 = cv2.initUndistortRectifyMap(
    cameraMatrix, 
    distCoeffs, 
    np.eye(3), 
    cameraMatrix, 
    (gi.width, gi.height),
      cv2.CV_16SC2
)

# defined constants
DETACHED_PROCESS = 0x00000008
BaseOutputDir = "./captures/"
SubCaptDir = "c_"
SaveFormat = ".jpg"
GrayCodeConverterPath = "../DecodeGrayImages/DecodeGrayImages"

# determines next capture folder index
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

# main capture loop
DoNextIteration = True
FirstIteration = True
while DoNextIteration:
    currentI += 1
    DoNextIteration = False
    CamDirOut = BaseOutputDir+SubCaptDir+str(currentI)+"/"

    # if there are images captured run the external decoder
    for imgnr in getImageIteration(FirstIteration, map1, map2):
        capture_and_save_image(CamDirOut+imgnr+SaveFormat)
        cv2.waitKey(1000)
        DoNextIteration=True
    if DoNextIteration:
        Popen(["python3 ConvertRawImage.py "+CamDirOut +" "+ SaveFormat +" "+ GrayCodeConverterPath], shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)
    FirstIteration=False
destroyW()
