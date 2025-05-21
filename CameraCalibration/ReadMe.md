This file serves as the ReadMe file for CameraCalibration.py
The file will show important information such as dependencies, libraries and flow of files graphically.

DEPENDENCIES
     Python3.12
     Numpy
     OS

DESCRIPTION
     BoardInfo.py -- Is used in KinectCameraCalibration.py, KinectStereoCalibration.py, GenerateAurcoAndChaurco.py, and CalibratePositionOfKinectDepthToCamera.py, this code defines board 
     layout, marker size, and dictionary settings for generating ArUco and ChArUco calibration patterns used in camera calibration scripts.
     
     CalibratePositionOfKinectDepthToCamera.py -- This code commits marker detection and corner extraction for camera calibration. Processing a set of already calibrated images by using          ArUco markers in Kinect RGB images and chessboard corners in the camera images. The script will generate visualization of markers and prepare data, saving the results in a designated        area for following steps.
     
     GenerateAurcoAndChaurco.py -- This code generates and saves .png files of the ChArUco board along with the ArUco board using Board.py. The images will then be used as reference 
     patterns for camera and projector calibration
     
     GetSecondViewPoints.py -- this file contains methods to estimate the projector coordinates based off the ChArUco corners in the cameras image. Computing a local homography around each 
     corner based on nearby points to map camera coordinates into projector coordinates, being essentialy for camera-projector calibration
     
     IRImageTestKinect.py -- this code uses OpenNI2 to interface with the kinect sensor to then capture and display both IR and color streams, showing until q is pressed
     
     KinectCameraCalibration.py
     
     KinectStereoCalibration.py
     
     main.py
     
     setup.py
     
     testcalibration.py

FILE DESCRIPTIONS


HOW TO RUN


FLOW CHART


FINAL COMMENTS
