# code creates a grayscale image with crosshairs at the center
# uses openCV to display
# to use simply run the code no arguments are required

import cv2
import numpy as np

# make a blank grayscale image (768x1024) initialized to zero (which is black)
img = np.zeros((768, 1024), dtype = np.uint8)
WINDOW_NAME="crosshairs"

# draws a thick horizontal line at the vertical center of the image
img[768//2, 1024//2-20:1024//2+20] = 255
img[768//2+1, 1024//2-20:1024//2+20] = 255

# draws a thick vertical line at the horizontal center of the image
img[768//2-20:768//2+20, 1024//2] = 255
img[768//2-20:768//2+20, 1024//2+1] = 255

# makes a smaller cross offset from center (possibly targeting marker)
img[768//2, 1024//2-20+150:1024//2+20-150] = 255
img[768//2+1, 1024//2-20+150:1024//2+20-150] = 255
img[768//2-20:768//2+20, 1024//2-150] = 255
img[768//2-20:768//2+20, 1024//2-151] = 255

# (optional line does not affect display since img is shown instead)
imgToDisplay = cv2.imread("binaryPatturn.png", cv2.IMREAD_GRAYSCALE)

# display the crosshairs image in a resizable window
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.imshow(WINDOW_NAME, img)
cv2.waitKey(0)

# sets the window to fullscreen and show again
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow(WINDOW_NAME, img)
cv2.waitKey(0)

# clean
cv2.destroyAllWindows()
