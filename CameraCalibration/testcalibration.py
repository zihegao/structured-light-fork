# System information:
# - Linux Mint 18.1 Cinnamon 64-bit
# - Python 2.7 with OpenCV 3.2.0

import numpy
import cv2
from cv2 import aruco
import pickle
import glob

dpmm = 40
A4_shape = 216, 279
A4_shape_margin = A4_shape[0]-10, A4_shape[1] - 20
outshape = A4_shape[0]*dpmm, A4_shape[1]*dpmm
desired_block_size_mm = 40
desired_aurco_size_mm = 21
desired_gap_size_mm = 5
 
blocksx = 9 # A4_shape[0]//desired_block_size_mm
blocksy = 7 # A4_shape[1]//desired_block_size_mm

# ChAruco board variables
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

# Create constants to be passed into OpenCV and Aruco methods
CHARUCO_BOARD = aruco.CharucoBoard((blocksx,
                                         blocksy),
                                         desired_block_size_mm,
                                         desired_aurco_size_mm,
                                         ARUCO_DICT)

# Create the arrays and variables we'll use to store info like corners and IDs from images processed
corners_all = [] # Corners discovered in all images processed
ids_all = [] # Aruco ids corresponding to corners discovered
image_size = None # Determined at runtime


# This requires a set of images or a video taken with the camera you want to calibrate
# I'm using a set of images taken with the camera with the naming convention:
# 'camera-pic-of-charucoboard-<NUMBER>.jpg'
# All images used should be the same size, which if taken with the same camera shouldn't be a problem
images = glob.glob('../captures/CharucoPhotoAngles/**/IMG*  .jpg', recursive=True)

# Loop through images glob'ed
for iname in images:
    # Open the image
    img = cv2.imread(iname)
    # Grayscale the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # Find aruco markers in the query image
    corners, ids, rejected = aruco.detectMarkers(
            image=gray,
            dictionary=ARUCO_DICT)

    # Outline the aruco markers found in our query image
    img2 = aruco.drawDetectedMarkers(
            image=img, 
            corners=corners,
            ids=ids)
    cv2.imshow('Display test' , img2)
    cv2.waitKey(0)


    # Get charuco corners and ids from detected aruco markers
    response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=img,
            board=CHARUCO_BOARD)

    # If a Charuco board was found, let's collect image/corner points
    # Requiring at least 20 squares
    if response > 20:
        # Add these corners and ids to our calibration arrays
        corners_all.append(charuco_corners)
        ids_all.append(charuco_ids)
        
        # Draw the Charuco board we've detected to show our calibrator the board was properly detected
        img = aruco.drawDetectedCornersCharuco(
                image=img,
                charucoCorners=charuco_corners,
                charucoIds=charuco_ids)
        
        # If our image size is unknown, set it now
        if not image_size:
            image_size = gray.shape[::-1]
    
        # Reproportion the image, maxing width or height at 1000
        proportion = max(img.shape) / 1000.0
        img = cv2.resize(img, (int(img.shape[1]/proportion), int(img.shape[0]/proportion)))
        # Pause to display each image, waiting for key press
        cv2.imshow('Charuco board', img)
        cv2.waitKey(0)
    else:
        print("Not able to detect a charuco board in image: {}".format(iname))

# Destroy any open CV windows
cv2.destroyAllWindows()

# Make sure at least one image was found
if len(images) < 1:
    # Calibration failed because there were no images, warn the user
    print("Calibration was unsuccessful. No images of charucoboards were found. Add images of charucoboards and use or alter the naming conventions used in this file.")
    # Exit for failure
    exit()

# Make sure we were able to calibrate on at least one charucoboard by checking
# if we ever determined the image size
if not image_size:
    # Calibration failed because we didn't see any charucoboards of the PatternSize used
    print("Calibration was unsuccessful. We couldn't detect charucoboards in any of the images supplied. Try changing the patternSize passed into Charucoboard_create(), or try different pictures of charucoboards.")
    # Exit for failure
    exit()

# Now that we've seen all of our images, perform the camera calibration
# based on the set of points we've discovered
calibration, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
        charucoCorners=corners_all,
        charucoIds=ids_all,
        board=CHARUCO_BOARD,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None)
    
# Print matrix and distortion coefficient to the console
print(cameraMatrix)
print(distCoeffs)
    
# Save values to be used where matrix+dist is required, for instance for posture estimation
# I save files in a pickle file, but you can use yaml or whatever works for you
f = open('calibration.pkl', 'wb')

d = { "cameraMatrix" : cameraMatrix, "distCoeffs" : distCoeffs, "rvecs": rvecs, "tvecs" : tvecs}
pickle.dump(d, f)
f.close()

#reload object from file
file2 = open('calibration.pkl', 'rb')
calibration = pickle.load(file2)
file2.close()


class calibrationobject:
    cameraMatrix = cameraMatrix
    distCoeffs = distCoeffs
    rvecs = rvecs
    tvecs = tvecs

calibration = calibrationobject()

pickle.dump(calibration, f)
f.close()
    
# Print to console our success
print('Calibration successful. Calibration file used: {}'.format('calibration.pckl'))