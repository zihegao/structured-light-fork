# CaptureCode.py
# ─────────────────────────────────────────────────────────────────────────────
# This script captures structured light patterns using a projector and camera
# It supports both Gray Code and Phase Shift pattern strategies
# The images are saved in folders like ./captures/<your-folder>/ for decoding later, chaarucotest from the tutorial
# Ensure camera and projector are correctly set up before running
# 
# Gray Code capture uses window utilities to step through patterns
# Phase Shift capture generates sinusoidal patterns, decodes wrapped phases
# and saves correspondence maps for later calibration
# ─────────────────────────────────────────────────────────────────────────────

from GrayCodesWindow import getImageIteration, destroyW
from CaptureImage import SaveImage, capture_and_save_image
import os
import numpy as np
from subprocess import Popen
import cv2
import time
import structuredlight as sl

# Windows-specific setting for subprocesses (not used in this version)
DETACHED_PROCESS = 0x00000008 

# Configuration and setup information, some is hardcoded so pay attention to values such as width/height
BaseOutputDirBeforeNew = "./captures/"
SubCaptDir = "c_"               # Capture folder prefix (e.g., c_0, c_1, ...)
SaveFormat = ".jpg"            # Format for saved images
WINDOW_NAME = "Projected Structured Light"  # Window name for projector display
num_fringes = 3                 # Number of fringe patterns per direction (must be >= 3)

PhaseShift = False              # Set True to use Phase Shift capture, False for Gray Code
width = 1920                   # Projector width (px)
height = 1080                 # Projector height (px)

# ─────────────────────────────────────────────────────────────────────────────
# Function: imShowAndCapture
# Shows an image on the projector and captures the corresponding camera image
# Inputs:
#   - cap: cv2.VideoCapture object
#   - img_pattern: numpy image to be projected
#   - delay: time to wait before capture (ms)
# Output:
#   - Grayscale captured camera image
# ─────────────────────────────────────────────────────────────────────────────
def imShowAndCapture(cap, img_pattern, delay=2000):
    cv2.imshow(WINDOW_NAME, img_pattern)
    cv2.waitKey(delay)
    ret, img_frame = cap.read()
    img_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)    
    return img_gray

# ─────────────────────────────────────────────────────────────────────────────
# Function: SaveImageCV
# Saves a list of images with incrementing filenames
# Inputs:
#   - images: list of images
#   - filename: base filename (e.g., 'h')
# Output:
#   - Files like h0.jpg, h1.jpg, h2.jpg saved
# ─────────────────────────────────────────────────────────────────────────────
def SaveImageCV(images, filename):
    i = 0
    for img in images:
        cv2.imwrite(filename + str(i) + SaveFormat, img)
        i += 1

# ─────────────────────────────────────────────────────────────────────────────
# Function: loadImages
# Loads a list of images based on base filename
# Inputs:
#   - filename: base filename
# Output:
#   - List of loaded images
# ─────────────────────────────────────────────────────────────────────────────
def loadImages(filename):
    img_list = []
    for i in range(num_fringes):
        img_list.append(cv2.imread(filename + str(i) + SaveFormat))
    return img_list

# Prompt user to name this capture session (folder inside ./captures/)
cFolder = input("Enter capture folder: ") 
BaseOutputDir = BaseOutputDirBeforeNew + cFolder + "/"

# Determine latest set index if folder already exists
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
else:
    os.makedirs(BaseOutputDir)
    print("making directory for "+ cFolder)

# Control flow for capture loop
DoNextIteration = True
FirstIteration = True

# ─────────────────────────────────────────────────────────────────────────────
# Main capture loop: allows capturing multiple sets (c_0, c_1, ...)
# ─────────────────────────────────────────────────────────────────────────────
while DoNextIteration:
    currentI += 1
    DoNextIteration = False
    CamDirOut = BaseOutputDir

    # ─────────────────────────────────────────────────────────────────────────
    # GRAY CODE MODE: Uses binary window sequence and manual camera triggers
    # ─────────────────────────────────────────────────────────────────────────
    if PhaseShift is False:
        for imgnr in getImageIteration(FirstIteration):
            capture_and_save_image(CamDirOut + imgnr + SaveFormat)
            DoNextIteration = True  # Keep looping if user continues
        destroyW()

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE SHIFT MODE: Uses structured sine patterns and auto decoding
    # ─────────────────────────────────────────────────────────────────────────
    if PhaseShift is True:
        cap = cv2.VideoCapture(0)

        # Wake camera
        _ = cap.read()
        ps = sl.PhaseShifting(num=num_fringes)

        # Display fullscreen black screen to initialize
        imgToDisplay = np.zeros((height, width), dtype=np.uint8)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.moveWindow(WINDOW_NAME, 900, -900)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(WINDOW_NAME, imgToDisplay)
        cv2.waitKey(2000)

        # Capture black and white reference frames
        imlist_b_img = imShowAndCapture(cap, imgToDisplay)
        cv2.imwrite(BaseOutputDir + "b" + SaveFormat, imlist_b_img) 
        imgToDisplay[:, :] = 255
        imlist_w_img = imShowAndCapture(cap, imgToDisplay)
        cv2.imwrite(BaseOutputDir + "w" + SaveFormat, imlist_w_img) 

        # Generate and capture horizontal fringe patterns
        imlist_posi_x_pat = ps.generate((width, height))
        imlist_posi_x_cap = [imShowAndCapture(cap, img) for img in imlist_posi_x_pat]  

        # Generate and capture vertical fringe patterns
        imlist = ps.generate((height, width))
        imlist_posi_y_pat = sl.transpose(imlist)
        imlist_posi_y_cap = [imShowAndCapture(cap, img) for img in imlist_posi_y_pat]

        # Save captured phase images
        SaveImageCV(imlist_posi_x_cap, BaseOutputDir + "h")
        SaveImageCV(imlist_posi_y_cap, BaseOutputDir + "v")

        # Decode phase index maps (wrapped)
        img_index_x = ps.decode(imlist_posi_x_cap)
        img_index_y = ps.decode(imlist_posi_y_cap)

        # Get pixel-to-projector correspondences
        campoints, prjpoints = sl.getCorrespondencePoints(img_index_x, img_index_y)
        print("xy-coord only")
        print("campoints: ", campoints.shape)
        print(campoints)
        print("prjpoints: ", prjpoints.shape)
        print(prjpoints)

        # Visualize and save correspondence maps
        img_correspondence_x = np.clip(img_index_x/width*255.0, 0, 255).astype(np.uint8)
        cv2.imshow("x_correspondence map", img_correspondence_x)
        cv2.imwrite(BaseOutputDir+"x_correspondence.png", img_correspondence_x)

        img_correspondence_y = np.clip(img_index_y/height*255.0, 0, 255).astype(np.uint8)
        cv2.imshow("y_correspondence map", img_correspondence_y)
        cv2.imwrite(BaseOutputDir+"y_correspondence.png", img_correspondence_y)

        img_correspondence = cv2.merge([
            0.0 * np.zeros_like(img_index_x),  # Blue channel unused
            img_index_x / width,              # Green channel: x
            img_index_y / height              # Red channel: y
        ])
        img_correspondence = np.clip(img_correspondence * 255.0, 0, 255).astype(np.uint8)
        cv2.imshow("x:Green, y:Red", img_correspondence)
        cv2.imwrite(BaseOutputDir+"x_ycorrespondence.png", img_correspondence)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    FirstIteration = False
