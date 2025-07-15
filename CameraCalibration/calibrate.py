# Camera Calibration using ChArUco Boards and Gray Code (or any) Structured Light
# Detects ChArUco corners in images, maps them to projector coordinates using Gray Code (or any structured light) decoding,
# and performs stereo calibration between a camera and a projector. Outputs include intrinsic and extrinsic matrices.

import os
import glob
import cv2
from calibration_board import board
import numpy as np
from cv2 import aruco
from GetSecondViewPoints import getCameraCoordinates

# path to calibration data
data_dir = """./captures/Calib3/"""   
output_dir = """./camera_calibration_out/"""
imgfmt = ".jpg"
projector_resolution =(1920, 1080)

dirnames = sorted(glob.glob(data_dir+"c_*"))

# storage for calibration points for both camera and projector
all_charuco_corners_camera = []
all_charco_corners_camera_2 = []
all_charco_corners_projector = []
all_charuco_ids_camera = []
all_charco_ids_projector = []
all_real_points = []

# process each calibration set
for dirname in dirnames:
    # Construct output path by replacing basePath root with ./camera_calibration_out/ and keeping the rest of the folder structure
    outPath = os.path.join("./camera_calibration_out", os.path.relpath(dirname, data_dir))
    os.makedirs(outPath, exist_ok=True)

    # load reference image
    img = cv2.imread(dirname+"/w"+imgfmt)

    if img is None:
        print("Skipping: "+dirname)
        continue

    # load Gray code decoded images and validity maps
    validV = cv2.imread(dirname+"/out_InvalidImageV.tiff", cv2.IMREAD_GRAYSCALE)
    validH = cv2.imread(dirname+"/out_InvalidImageH.tiff", cv2.IMREAD_GRAYSCALE)
    ### why switched V and H?
    coordsV = cv2.imread(dirname+"/out_BinImageV.tiff", cv2.IMREAD_ANYDEPTH+cv2.IMREAD_GRAYSCALE)
    coordsH = cv2.imread(dirname+"/out_BinImageH.tiff", cv2.IMREAD_ANYDEPTH+cv2.IMREAD_GRAYSCALE)

    ######### Aruco marker and Charuco corner detection #########
    # detect Aruco markers
    corners, ids, rejected = aruco.detectMarkers(img, board.getDictionary())

    # try to recover Aruco markers based on board info if they are not directly detected
    if len(corners) > 0:
        corners, ids, rejected, recovered = aruco.refineDetectedMarkers(
            img, board, corners, ids, rejected)

    # optionally visualize detected Aruco markers
    cimg = aruco.drawDetectedMarkers(img.copy(), corners, ids)
    cv2.imwrite(outPath+"/DetectedMarkers.png", cimg)

    # interpolate ChArUco corners using ArUco markers and board info
    charucoCorners, charucoIds = [], []
    if len(ids) > 0:
        numCorners, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(corners, ids, img, board)
    if charucoIds is None:
        continue # skip this image entirely if no charuco corners are found

    # optionally visualize detected charuco corners
    cimg = aruco.drawDetectedCornersCharuco(img.copy(), charucoCorners, charucoIds)
    cv2.imwrite(outPath+"/DetectedCorners.png", cimg)

    # saves detected charucocorners and IDs
    all_charuco_corners_camera.append(charucoCorners.copy())
    all_charuco_ids_camera.append(charucoIds.copy())


    # get corresponding projector and filtered camera cordinates
    valid_points, new_points_cam, new_points_projector = getCameraCoordinates(img, validV, validH, coordsV, coordsH, charucoCorners)
    charucoIds = charucoIds[valid_points]

    if charucoIds is None:
        continue

    # saves filtered points for stereo calibration
    all_charco_corners_camera_2.append(new_points_cam)
    print(charucoIds[:, 0])
    all_real_points.append(board.getChessboardCorners())
    print(all_real_points)

    print(new_points_projector)
    all_charco_corners_projector.append(new_points_projector)
    all_charco_ids_projector.append(charucoIds)

# save resolution of the camera image   
camera_resolution = img.shape[:-1]

#CalibrationFlags=cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3
rep_err_camera, mtx_camera, dist_camera, rvecs_camera, tvecs_camera = cv2.aruco.calibrateCameraCharuco(all_charuco_corners_camera, all_charuco_ids_camera, board, camera_resolution, None, None, flags=cv2.CALIB_FIX_K2+cv2.CALIB_FIX_K3+cv2.CALIB_FIX_K4+cv2.CALIB_FIX_K5+cv2.CALIB_FIX_K6)
rep_err_proj, mtx_proj, dist_proj, rvecs_proj, tvecs_proj = cv2.aruco.calibrateCameraCharuco(all_charco_corners_projector, all_charco_ids_projector, board, projector_resolution, None, None, flags=cv2.CALIB_FIX_K2+cv2.CALIB_FIX_K3+cv2.CALIB_FIX_K4+cv2.CALIB_FIX_K5+cv2.CALIB_FIX_K6)

#np.savez("../camera_calibration_out/calculated_cams_matrix.npz", rep_err_camera=rep_err_camera, mtx_camera=mtx_camera, dist_camera=dist_camera , rvecs_camera=rvecs_camera, tvecs_camera=tvecs_camera, newcameramtx_camera=newcameramtx_camera, roi_camera=newcameramtx_camera,
 #        rep_err_proj=rep_err_proj, mtx_proj=mtx_proj, dist_proj=dist_proj, rvecs_proj=rvecs_proj, tvecs_proj=tvecs_proj, newcameramtx_proj=newcameramtx_proj, roi_proj=newcameramtx_proj)

retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
    cv2.stereoCalibrate(all_real_points, all_charco_corners_camera_2, all_charco_corners_projector,
                        mtx_camera, dist_camera, mtx_proj,
                        dist_proj, camera_resolution, flags=cv2.CALIB_FIX_INTRINSIC)

newcameramtx_camera, roi_camera=cv2.getOptimalNewCameraMatrix(cameraMatrix1,distCoeffs1,camera_resolution,1, camera_resolution)
newcameramtx_proj, roi_proj=cv2.getOptimalNewCameraMatrix(cameraMatrix2,distCoeffs2,projector_resolution,1, projector_resolution)

invCamMtx = np.linalg.inv(newcameramtx_camera)
invProjMtx = np.linalg.inv(newcameramtx_proj)

# save less distorted calibration results
np.savez(output_dir+"/calculated_cams_matrix_less_distortion.npz",
         retval=retval,
         cameraMatrix1=cameraMatrix1,
         distCoeffs1=distCoeffs1,
         cameraMatrix2=cameraMatrix2,
         distCoeffs2=distCoeffs2,
         R=R,
         T=T,
         E=E,
         F=F,
         newcameramtx_camera=newcameramtx_camera,
         roi_camera=roi_camera,
         newcameramtx_proj=newcameramtx_proj,
         roi_proj=roi_proj,
         invCamMtx=invCamMtx,
         invProjMtx=invProjMtx)

# repeat calibration *without* distortion fix flags (more flexible, higher potential distortion)
rep_err_camera, mtx_camera, dist_camera, rvecs_camera, tvecs_camera = cv2.aruco.calibrateCameraCharuco(all_charuco_corners_camera, all_charuco_ids_camera, board, camera_resolution, None, None)
rep_err_proj, mtx_proj, dist_proj, rvecs_proj, tvecs_proj = cv2.aruco.calibrateCameraCharuco(all_charco_corners_projector, all_charco_ids_projector, board, projector_resolution, None, None)

retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
    cv2.stereoCalibrate(all_real_points, all_charco_corners_camera_2, all_charco_corners_projector,
                        mtx_camera, dist_camera, mtx_proj,
                        dist_proj, camera_resolution, flags=cv2.CALIB_FIX_INTRINSIC)

# yet again, compute optimal undistorted matrices
newcameramtx_camera, roi_camera=cv2.getOptimalNewCameraMatrix(cameraMatrix1,distCoeffs1,camera_resolution,1, camera_resolution)
newcameramtx_proj, roi_proj=cv2.getOptimalNewCameraMatrix(cameraMatrix2,distCoeffs2,projector_resolution,1, projector_resolution)

invCamMtx = np.linalg.inv(newcameramtx_camera)
invProjMtx = np.linalg.inv(newcameramtx_proj)

# save the full (possibly more distorted) calibration result
np.savez(output_dir+"/calculated_cams_matrix.npz",
         retval=retval,
         cameraMatrix1=cameraMatrix1,
         distCoeffs1=distCoeffs1,
         cameraMatrix2=cameraMatrix2,
         distCoeffs2=distCoeffs2,
         R=R,
         T=T,
         E=E,
         F=F,
         newcameramtx_camera=newcameramtx_camera,
         roi_camera=roi_camera,
         newcameramtx_proj=newcameramtx_proj,
         roi_proj=roi_proj,
         invCamMtx=invCamMtx,
         invProjMtx=invProjMtx)

print("Camera calibration results:")
print(f"Camera matrix:\n{cameraMatrix1}")           
print(f"Distortion coefficients:\n{distCoeffs1}")
print(f"New camera matrix:\n{newcameramtx_camera}")
print(f"ROI for camera: {roi_camera}")
print(f"Camera reprojection error: {rep_err_camera}")  # Added line

print()
print("Projector calibration results:")
print(f"Projector matrix:\n{cameraMatrix2}")
print(f"Distortion coefficients:\n{distCoeffs2}")
print(f"New projector matrix:\n{newcameramtx_proj}")
print(f"ROI for projector: {roi_proj}")
print(f"Projector reprojection error: {rep_err_proj}") 


