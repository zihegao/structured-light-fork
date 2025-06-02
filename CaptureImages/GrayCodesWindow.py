# code displays a sequence of Gray code pattern images using OpenCV
# utilizes GrayImage class (from GrayImages.py) optionally applies remapping
# press Enter when tab on other screen to start the fullscreen slideshow of encoded bit planes, then wait for process to finish

import cv2
import numpy as np
from GrayImages import GrayImage

gImg = GrayImage()
WINDOW_NAME = "GrayCodesWindow"

# disp Gray code in window, begins with prompt then iterates through all images
def getImageIteration(firstIteration=True, map1=None, map2=None):
    if firstIteration:
        print("First It")
        imgToDisplay = cv2.imread("grayPatturn.png", cv2.IMREAD_GRAYSCALE)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        #cv2.imshow(WINDOW_NAME, imgToDisplay)
        #cv2.resizeWindow(WINDOW_NAME, imgToDisplay.shape[1], imgToDisplay.shape[0])

        # displays the highest bit Gray code pattern
        imgToDisplay = gImg.getImage(gImg.num_bits-1)
        cv2.imshow(WINDOW_NAME, imgToDisplay)
    else:
        print("Other Its")
        imgToDisplay = np.array([[0]], dtype=np.uint8)
        cv2.imshow(WINDOW_NAME, imgToDisplay)

    # waits for the key press
    k = cv2.waitKey(0)

    # must be the enter (return) key
    if k != ord('\r'):
        return
    
    if firstIteration:
        #make full screen
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);
    # pause before starting sideshow
    k = cv2.waitKey(2000)
    for imgnr, imgToDisplay in gImg.getIterator():
        print(imgnr)

        # aply remapping if calibration maps are given
        if map1 is not None and map2 is not None:
            imgToDisplay = cv2.remap(imgToDisplay, map1, map2, cv2.INTER_NEAREST)

        # display the current Gray code image
        cv2.imshow(WINDOW_NAME, imgToDisplay)
        cv2.waitKey(10)
        yield imgnr

def destroyW():
    cv2.destroyWindow(WINDOW_NAME)
