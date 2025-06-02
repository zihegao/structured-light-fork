# captures images when using a projector and camera.
# currently can do either
#    Gray Code Pattern Capture
#    Phase Shift Pattern Capture

# important notes before running
# ensure projector is set correctly on the second screen
# adjust `width`, `height`, and camera index as specified by own setup
# Required modules: OpenCV, NumPy, structuredlight, which are demonstrated how to download in the Structured Light Procedures.pdf


from GrayCodesWindow import getImageIteration, destroyW
from CaptureImage import SaveImage, capture_and_save_image
import os
import numpy as np
from subprocess import Popen
import cv2

import time
import structuredlight as sl

DETACHED_PROCESS = 0x00000008 
BaseOutputDirBeforeNew = "./captures/"
SubCaptDir = "c_"
SaveFormat = ".jpg"
WINDOW_NAME="Projected Structured Light"
num_fringes = 3 # cant be less than 3

# ATTENTION, important for if phase shift or grey code
PhaseShift = False

# parameters attuned to own setup
width = 1920
height = 1080

# displays the specified pattern and captures the image
def imShowAndCapture(cap, img_pattern, delay=1000):
    cv2.imshow(WINDOW_NAME, img_pattern)
    cv2.waitKey(delay)
    ret, img_frame = cap.read()
    img_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)    
    return img_gray

# save image to disk while incrementing names
def SaveImageCV(images, filename):
            i=0
            for img in images:
                cv2.imwrite(filename + str(i) + SaveFormat, img)
                i+= 1

# load the previous image ased upon the numerical pattern
def loadImages(filename):
    img_list = []
    for i in range(num_fringes):
        img_list.append( cv2.imread(filename + str(i) + SaveFormat))
    return img_list


# initialize/ask for area on where to store images
GrayCodeConverterPath = "../DecodeGrayImages/DecodeGrayImages"
cFolder = input("Enter capture folder: ") 
# cFolder = "newphaseshift"
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
if os.path.isdir(BaseOutputDir) == False:
    os.makedirs(BaseOutputDir)
    print("making directory for "+ cFolder)
DoNextIteration = True
FirstIteration = True


while DoNextIteration:
    currentI += 1
    DoNextIteration = False
    # CamDirOut = BaseOutputDir+SubCaptDir+str(currentI)+"/"
    CamDirOut = BaseOutputDir

    # grey code mode
    if PhaseShift is False:
        for imgnr in getImageIteration(FirstIteration):
            #SaveImage(CamDirOut+imgnr+SaveFormat)
            capture_and_save_image(CamDirOut+imgnr+SaveFormat)

            DoNextIteration=True
        destroyW()

    # phase shift mode
    if PhaseShift is True:
        cap = cv2.VideoCapture(0)

        # takes image to wake camera
        testcap = cap.read()
        ps = sl.PhaseShifting(num=num_fringes)

        imgToDisplay = np.zeros((height, width), dtype = np.uint8)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.moveWindow(WINDOW_NAME, 900,-900)
        cv2.imshow(WINDOW_NAME, imgToDisplay)
        # cv2.waitKey(0)

        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(WINDOW_NAME, imgToDisplay)
        cv2.waitKey(2000)

        # capture black and white images for refernce
        imlist_b_img = imShowAndCapture(cap, imgToDisplay)
        cv2.imwrite(BaseOutputDir + "b" + SaveFormat, imlist_b_img) 
        imgToDisplay[:,:] = 255
        imlist_w_img = imShowAndCapture(cap, imgToDisplay)
        cv2.imwrite(BaseOutputDir + "w" + SaveFormat, imlist_w_img) 
        
        # Generate and Decode x-coord
        # Generate
        imlist_posi_x_pat = ps.generate((width, height))
        # Capture
        imlist_posi_x_cap = [imShowAndCapture(cap, img) for img in imlist_posi_x_pat]  
        
        imlist = ps.generate((height, width))

        # generate and capture y fringe patterns
        imlist_posi_y_pat = sl.transpose(imlist)
        imlist_posi_y_cap = [ imShowAndCapture(cap, img) for img in imlist_posi_y_pat]

        # now save the images
        SaveImageCV(imlist_posi_x_cap, BaseOutputDir + "h")
        SaveImageCV(imlist_posi_y_cap, BaseOutputDir + "v")

        # decode and visualize 
        img_index_x = ps.decode(imlist_posi_x_cap)
        img_index_y = ps.decode(imlist_posi_y_cap)

        campoints, prjpoints = sl.getCorrespondencePoints(img_index_x, img_index_y)

        print("xy-coord only")
        print("campoints: ", campoints.shape)
        print(campoints)
        print("prjpoints: ", prjpoints.shape)
        print(prjpoints)

        img_correspondence_x = np.clip(img_index_x/width*255.0, 0, 255).astype(np.uint8)
        cv2.imshow("x_corresponnence map", img_correspondence_x)
        cv2.imwrite(BaseOutputDir+"x_correspondence.png", img_correspondence_x)

        img_correspondence_y = np.clip(img_index_y/width*255.0, 0, 255).astype(np.uint8)
        cv2.imshow("y_corresponnence map", img_correspondence_y)
        cv2.imwrite(BaseOutputDir+"y_correspondence.png", img_correspondence_y)

        img_correspondence = cv2.merge([0.0*np.zeros_like(img_index_x), img_index_x/width, img_index_y/height])
        img_correspondence = np.clip(img_correspondence*255.0, 0, 255).astype(np.uint8)
        
        cv2.imshow("x:Green, y:Red", img_correspondence)
        cv2.imwrite(BaseOutputDir+"x_ycorrespondence.png", img_correspondence)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # if DoNextIteration:
    #     Popen(["python", "CaptureImages\ConvertRawImage.py", "+CamDirOut", "SaveFormat", "GrayCodeConverterPath"], shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)

    FirstIteration=False
