# Gray Code Camera-Projector Calibration Script
# ─────────────────────────────────────────────────────────────────────────────
# This script performs camera-projector stereo calibration using structured light.
# It uses ChArUco boards for camera corner detection and Gray code projection
# to map projector pixel coordinates. With this, it computes intrinsic and extrinsic
# calibration for both devices using OpenCV stereo calibration tools.
# Outputs are stored in .npz files for later use in 3D reconstruction.
# ─────────────────────────────────────────────────────────────────────────────

import os
import cv2
import BoardInfo
import numpy as np
from cv2 import aruco
from GetSecondViewPoints import getCameraCoordinates

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# Specify which calibration folders (e.g. c_0, c_1, ...) to use
# ─────────────────────────────────────────────────────────────────────────────
directories_to_use = [0]  # Can be extended to [0,1,2,...] for more robust calibration

# Set up template paths
basePath = "./captures/Calib3/c_{0}/"                    # Input image folder
outPathTemplate = "./camera_calibration_out/Calib3/c_{0}/"  # Output for intermediate results
imgfmt = ".jpg"                                          # Image file format
projector_resolution = (1920, 1080)                      # Projector pixel resolution

# Initialize data storage for corner detections and IDs
all_charco_corners_camera = []       # Raw detected ChArUco corners in camera images
all_charco_corners_camera_2 = []     # Valid camera corners with projector mapping
all_charco_corners_projector = []    # Corresponding projector coordinates
all_charco_ids_camera = []           # Corner IDs from camera
all_charco_ids_projector = []        # Corner IDs for projector
all_real_points = []                 # Known 3D coordinates from board (Z = 0)

# ─────────────────────────────────────────────────────────────────────────────
# PROCESS EACH CALIBRATION FOLDER (e.g., c_0, c_1, ...)
# ─────────────────────────────────────────────────────────────────────────────
for dirnum in directories_to_use:
    path = basePath.format(dirnum)
    outPath = outPathTemplate.format(dirnum)
    os.makedirs(outPath, exist_ok=True)

    # Load the white frame used for ArUco/ChArUco detection
    img = cv2.imread(path + "w" + imgfmt)
    if img is None:
        print("Skipping: " + path)
        continue

    # Load phase-based decoding results from projector side
    validV = cv2.imread(path + "out_InvalidImageV.tiff", cv2.IMREAD_GRAYSCALE)  # invalid pixels (vertical)
    validH = cv2.imread(path + "out_InvalidImageH.tiff", cv2.IMREAD_GRAYSCALE)  # invalid pixels (horizontal)
    coordsV = cv2.imread(path + "out_BinImageH.tiff", cv2.IMREAD_ANYDEPTH + cv2.IMREAD_GRAYSCALE)  # projector x-coord
    coordsH = cv2.imread(path + "out_BinImageV.tiff", cv2.IMREAD_ANYDEPTH + cv2.IMREAD_GRAYSCALE)  # projector y-coord

    # Detect ArUco markers in white image (gives 2D camera space corners)
    corners, ids, rejected = aruco.detectMarkers(img, BoardInfo.arucoDict)
    cimg = aruco.drawDetectedMarkers(img.copy(), corners, ids)
    cv2.imwrite(outPath + "DetectedMarkers.png", cimg)

    # Interpolate subpixel corners inside detected ArUco squares
    charucoCorners, charucoIds = [], []
    if len(ids) > 0:
        _, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(
            corners, ids, img, BoardInfo.charucoBoard)
    if charucoIds is None:
        continue  # skip if no valid corners found

    # Save raw camera corners and IDs for intrinsic calibration
    all_charco_corners_camera.append(charucoCorners.copy())
    all_charco_ids_camera.append(charucoIds.copy())

    # Visualize interpolated corners and save image
    cimg = aruco.drawDetectedCornersCharuco(img.copy(), charucoCorners, charucoIds)
    cv2.imwrite(outPath + "DetectedCorners.png", cimg)

    # Get valid 2D matches: map camera corners to projector coordinates using Gray code
    valid_points, new_points_cam, new_points_projector = getCameraCoordinates(
        img, validV, validH, coordsV, coordsH, charucoCorners)

    # Filter IDs to keep only those with valid projector correspondences
    charucoIds = charucoIds[valid_points]
    if charucoIds is None:
        continue

    # Store filtered calibration points and projector correspondences
    all_charco_corners_camera_2.append(new_points_cam)
    all_charco_corners_projector.append(new_points_projector)
    all_charco_ids_projector.append(charucoIds)

    # All real world points from the CharUco board (Z=0 plane)
    all_real_points.append(BoardInfo.charucoBoard.getChessboardCorners())

    # Optional debug output
    print(charucoIds[:, 0])
    print(new_points_projector)

# Get the image resolution for camera (used in calibration)
camera_resolution = img.shape[:-1]

# ─────────────────────────────────────────────────────────────────────────────
# FIRST PASS CALIBRATION (with distortion constraints)
# Intrinsics for both camera and projector
# ─────────────────────────────────────────────────────────────────────────────
rep_err_camera, mtx_camera, dist_camera, _, _ = cv2.aruco.calibrateCameraCharuco(
    all_charco_corners_camera,
    all_charco_ids_camera,
    BoardInfo.charucoBoard,
    camera_resolution,
    None, None,
    flags=cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6
)

rep_err_proj, mtx_proj, dist_proj, _, _ = cv2.aruco.calibrateCameraCharuco(
    all_charco_corners_projector,
    all_charco_ids_projector,
    BoardInfo.charucoBoard,
    projector_resolution,
    None, None,
    flags=cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6
)

# Stereo calibration (known intrinsics, solve for R/T)
retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
    all_real_points,
    all_charco_corners_camera_2,
    all_charco_corners_projector,
    mtx_camera, dist_camera, mtx_proj, dist_proj,
    camera_resolution,
    flags=cv2.CALIB_FIX_INTRINSIC
)

# Compute optimal undistorted projection matrices (rectification)
newcameramtx_camera, roi_camera = cv2.getOptimalNewCameraMatrix(cameraMatrix1, distCoeffs1, camera_resolution, 1, camera_resolution)
newcameramtx_proj, roi_proj = cv2.getOptimalNewCameraMatrix(cameraMatrix2, distCoeffs2, projector_resolution, 1, projector_resolution)

# Precompute inverse intrinsics for later 3D reprojection
invCamMtx = np.linalg.inv(newcameramtx_camera)
invProjMtx = np.linalg.inv(newcameramtx_proj)

# Save the first-pass (less distorted) calibration results
np.savez("./camera_calibration_out/calculated_cams_matrix_less_distortion.npz",
    retval=retval,
    cameraMatrix1=cameraMatrix1,
    distCoeffs1=distCoeffs1,
    cameraMatrix2=cameraMatrix2,
    distCoeffs2=distCoeffs2,
    R=R, T=T, E=E, F=F,
    newcameramtx_camera=newcameramtx_camera,
    roi_camera=roi_camera,
    newcameramtx_proj=newcameramtx_proj,
    roi_proj=roi_proj,
    invCamMtx=invCamMtx,
    invProjMtx=invProjMtx)

# ─────────────────────────────────────────────────────────────────────────────
# SECOND PASS CALIBRATION (with distortion enabled)
# Re-runs calibration without constraint flags to allow more flexibility
# May yield slightly higher distortion but better real-world fit
# ─────────────────────────────────────────────────────────────────────────────
rep_err_camera, mtx_camera, dist_camera, _, _ = cv2.aruco.calibrateCameraCharuco(
    all_charco_corners_camera, all_charco_ids_camera, BoardInfo.charucoBoard, camera_resolution, None, None)
rep_err_proj, mtx_proj, dist_proj, _, _ = cv2.aruco.calibrateCameraCharuco(
    all_charco_corners_projector, all_charco_ids_projector, BoardInfo.charucoBoard, projector_resolution, None, None)

# Stereo calibration again
retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
    all_real_points,
    all_charco_corners_camera_2,
    all_charco_corners_projector,
    mtx_camera, dist_camera, mtx_proj, dist_proj,
    camera_resolution,
    flags=cv2.CALIB_FIX_INTRINSIC)

# Undistortion again
newcameramtx_camera, roi_camera = cv2.getOptimalNewCameraMatrix(cameraMatrix1, distCoeffs1, camera_resolution, 1, camera_resolution)
newcameramtx_proj, roi_proj = cv2.getOptimalNewCameraMatrix(cameraMatrix2, distCoeffs2, projector_resolution, 1, projector_resolution)

invCamMtx = np.linalg.inv(newcameramtx_camera)
invProjMtx = np.linalg.inv(newcameramtx_proj)

# Save final calibration result
np.savez("./camera_calibration_out/calculated_cams_matrix.npz",
    retval=retval,
    cameraMatrix1=cameraMatrix1,
    distCoeffs1=distCoeffs1,
    cameraMatrix2=cameraMatrix2,
    distCoeffs2=distCoeffs2,
    R=R, T=T, E=E, F=F,
    newcameramtx_camera=newcameramtx_camera,
    roi_camera=roi_camera,
    newcameramtx_proj=newcameramtx_proj,
    roi_proj=roi_proj,
    invCamMtx=invCamMtx,
    invProjMtx=invProjMtx)
