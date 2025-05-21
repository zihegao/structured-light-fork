import cv2
from cv2 import aruco
import BoardInfo     # has the board setup information 
from GetSecondViewPoints import getCameraCoordinates
import os
import numpy as np

# ===== parameters ======

# directory indices to process
directories_to_use = [i for i in range(21)]

# created template paths for input images and output folders
basePath = """../captures/Calib3/c_{0}/"""
outPathTemplate = """../camera_calibration_out/Calib3/c_{0}/"""

# resolution of the current projector 
projector_resolution =(1920, 1080)

# ======= date storage values =====

# lists to hold ChArUco marker corner positions and IDs
all_charco_corners_camera = []
all_charco_corners_camera_2 = []
all_charco_corners_projector = []
all_charco_ids_camera = []
all_charco_ids_projector = []

# list to store 3D real-world corner locations
all_real_points = []

# === Main Processing Loop ===

for dirnum in directories_to_use:
    path = basePath.format(dirnum)
    outPath = outPathTemplate.format(dirnum)

     # confirms that the output directory exists
    os.makedirs(outPath, exist_ok=True)

     # loads RGB image (projector view) and Kinect RGB image (calibration view)
    img_camera = cv2.imread(path+"w.jpg")
    img_kinect = cv2.imread(path+"kinect___rgb.png")

    # skips if image couldn't be loaded to prevent deadlock/full stop
    if img_kinect is None:
        print("Skipping: "+path)
        continue

    # detect ArUco markers in the Kinect image
    corners, ids, rejected = aruco.detectMarkers(img_kinect, BoardInfo.aurcoDict)

    # visualize detected markers for debugging
    cimg = aruco.drawDetectedMarkers(img_kinect.copy(), corners, ids)

    # attempts to find chessboard corners in the RGB camera image
    retval, corners = cv2.findChessboardCorners(img_camera, (10,13))
    
    #cimg = cv2.drawChessboardCorners(img_camera.copy(), (10,13), corners, True)
    print(retval)

     # displays marker/chessboard result for manual inspection
    cv2.imshow("tset", cimg)
    cv2.waitKey(0)

