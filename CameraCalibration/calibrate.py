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

PATCH_SIZE = 47 # local homography patch size

# path to calibration data
data_dir = """./captures/Calib3/"""   
output_dir = """./camera_calibration_out/"""
imgfmt = ".jpg"
projector_shape =(1920, 1080)

dirnames = sorted(glob.glob(data_dir+"c_*"))

img_files_lists = [] # a list of lists, each containing image files for a specific calibration data folder
for dirname in dirnames:
    img_files = sorted((glob.glob(os.path.join(dirname, '*'+imgfmt))))
    if len(img_files) == 0:
        dirnames.remove(dirname)
        continue
    img_files_lists.append(img_files)

def joint_calibrate(data_dirs, 
              board, projector_shape, output_dir):
    """
    Function to calibrate camera and projector using ChArUco boards and decoded structured-light correspondence.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print('Calibrating ...')
    camera_shape = cv2.imread(img_files_lists[0][0], cv2.IMREAD_GRAYSCALE).shape

    # the following lists store calibration points for all board configrurations (frames)
    # each item in the list is a Nx1x2 array of 2D points (N is the number of detected corners in that configuration/frame)
    # len(list) = number of configurations/frames
    cam_corners_list = [] # charuco corners in camera coordinates
    charuco_ids_list = [] # charuco IDs
    proj_corners_list = [] # charuco corners in projector coordinates
    is_corner_jointly_valid_list = [] # if corner has both valid camera and projector coordinates
    obj_points_list = [] # known object points on the calibration board

    for dirname in data_dirs:
        # Construct output path by replacing basePath root with ./camera_calibration_out/ and keeping the rest of the folder structure
        outPath = os.path.join(output_dir, os.path.relpath(dirname, data_dir))
        os.makedirs(outPath, exist_ok=True)

        # load white (all pixels on) image to identify charuco corners
        img = cv2.imread(glob.glob(dirname+"/w"+imgfmt)[0], cv2.IMREAD_GRAYSCALE)

        if img is None:
            print("Skipping: "+dirname)
            continue

        # load Gray code decoded images and validity maps
        validV = cv2.imread(os.path.join(dirname, "out_InvalidImageV.tiff"), cv2.IMREAD_GRAYSCALE)
        validH = cv2.imread(os.path.join(dirname, "out_InvalidImageH.tiff"), cv2.IMREAD_GRAYSCALE)
        coordsV = cv2.imread(os.path.join(dirname, "out_BinImageV.tiff"), cv2.IMREAD_ANYDEPTH+cv2.IMREAD_GRAYSCALE)
        coordsH = cv2.imread(os.path.join(dirname, "out_BinImageH.tiff"), cv2.IMREAD_ANYDEPTH+cv2.IMREAD_GRAYSCALE)

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

        if charucoIds is None or len(charucoIds) == 0:
            continue
        numCorners, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(corners, ids, img, board)
        
        is_corner_valid, proj_corners = get_projector_corner_coords(img, validV, validH, coordsV, coordsH, charucoCorners, patch_size=PATCH_SIZE)
        # all_cam_corners.append()

def get_projector_corner_coords(img, validV, validH, coordsV, coordsH, cam_corners, patch_size=PATCH_SIZE):
    # From camera corner coordinates and decoded structured light coordinates,
    # compute projector corner coordinates using local homography 
    # (calculated from camera and projector coordinates in patches around each corner)
    # following the approach from the paper:
    # "Simple, Accurate, and Robust Projector-Camera Calibration" by Daniel Moreno and Gabriel Taubin

    # input: img (the image where corners are detected, for visualization purpose only)
    #      validV, validH (validity maps for vertical and horizontal Gray code decoding, 0 means valid)
    #      coordsV, coordsH (decoded structured-light coordinates/correspondences in vertical and horizontal directions)
    #     cam_corners (Nx1x2 array of detected corner coordinates in camera coordinates)
    #     patch_size (size of the local patch around each corner to use for homography computation)
    # output: is_corner_valid (list of bools indicating if each corner has valid projector coordinates)
    #      proj_corners (Nx1x2 array of detected corner coordinates in projector coordinates)
    proj_corners = [] 
    is_corner_valid = []

    # patch relative coordinates, creating a grid of coordinates around a point (patch_size x patch_size) centered at (0,0)
    patch_rel_coords = np.indices((patch_size, patch_size)).reshape(2, -1).T - (patch_size-1)//2

    for corner in cam_corners:

        # find coordinates of all points in the patch around the current corner
        patch_coords = (np.rint(corner[:]) + patch_rel_coords).astype(np.int32)
        patch_idx = patch_coords[:, [1, 0]] # indexing in image matrix is (i,j) = (y,x)

        # keep only points inside the image boundaries
        patch_idx = patch_idx[np.logical_and(np.logical_and(patch_idx[:, 0] < img.shape[0],
                                                                            patch_idx[:, 0] >= 0),
                                                             np.logical_and(patch_idx[:, 1] < img.shape[1],
                                                                            patch_idx[:, 1] >= 0))]
       
        # check if point in patch has valid projector decoding (not masked by validV and validH)
        isValid = np.logical_and(validV[patch_idx[:, 0], patch_idx[:, 1]] == 0,
                                 validH[patch_idx[:, 0], patch_idx[:, 1]] == 0)
        patch_idx = patch_idx[isValid]

        # get corresponding projector coordinates for valid points in patch
        proj_coords_u = coordsH[patch_idx[:, 0], patch_idx[:, 1]]
        proj_coords_v = coordsV[patch_idx[:, 0], patch_idx[:, 1]]
        proj_coords = np.stack((proj_coords_u, proj_coords_v), axis=1)

        # highlight these points on the image for visualization
        img[patch_idx[:, 0], patch_idx[:, 1]] = 255

        if len(proj_coords) < patch_size**2 * 0.3: # if less than 30% of the patch has valid projector coordinates, skip this corner 
            is_corner_valid.append(False)
            print(f'    Warning : corner {int(corner[0,0]), int(corner[0,1])} \
                  was skiped because decoded pixels were too few (check your images and threasholds)')
            continue

        # compute homography matrix from camera points to projector points
        H, inliers = cv2.findHomography(
            patch_coords, 
            proj_coords, 
            ransacReprojThreshold=2, maxIters=100000, method = cv2.FM_LMEDS, confidence=0.99
            )
        if H is None:
            is_corner_valid.append(False)
            print(f'    Warning: corner {int(corner[0,0]), int(corner[0,1])} \
                  was skipped because homography computation failed (check your images and thresholds)')

        # map the current corner point through the homography to get projector coordinates
        proj_corner = np.dot(H, [corner[0,1], corner[0,0], 1.0])
        proj_corner /= proj_corner[2] # homogeneous coordinate normalization
        proj_corners.append([[proj_corner[0], proj_corner[1]]])

    return is_corner_valid, np.array(proj_corners, dtype=np.float32)






# storage for calibration points for both camera and projector
all_charuco_corners_camera = []
all_charco_corners_camera_2 = []
all_charco_corners_projector = []
all_charuco_ids_camera = []
all_charco_ids_projector = []
all_obj_points = [] # known object points on the calibration board

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
    cimg = aruco.drawDetectedCornersCharuco(img, charucoCorners, charucoIds)
    cv2.imwrite(outPath+"/DetectedCorners.png", cimg)

    # saves detected charucocorners and IDs
    all_charuco_corners_camera.append(charucoCorners.copy())
    all_charuco_ids_camera.append(charucoIds.copy())


    # get corresponding projector and filtered camera cordinates
    valid_points, new_points_cam, proj_corners = getCameraCoordinates(img, validV, validH, coordsV, coordsH, charucoCorners)
    charucoIds = charucoIds[valid_points]

    if charucoIds is None:
        continue

    # saves filtered points for stereo calibration
    all_charco_corners_camera_2.append(new_points_cam)
    print(charucoIds[:, 0])
    all_obj_points.append(board.getChessboardCorners())
    print(all_obj_points)

    print(proj_corners)
    all_charco_corners_projector.append(proj_corners)
    all_charco_ids_projector.append(charucoIds)

# save resolution of the camera image   
camera_resolution = img.shape[:-1]

#CalibrationFlags=cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3
rep_err_camera, mtx_camera, dist_camera, rvecs_camera, tvecs_camera = cv2.aruco.calibrateCameraCharuco(all_charuco_corners_camera, all_charuco_ids_camera, board, camera_resolution, None, None, flags=cv2.CALIB_FIX_K2+cv2.CALIB_FIX_K3+cv2.CALIB_FIX_K4+cv2.CALIB_FIX_K5+cv2.CALIB_FIX_K6)
rep_err_proj, mtx_proj, dist_proj, rvecs_proj, tvecs_proj = cv2.aruco.calibrateCameraCharuco(all_charco_corners_projector, all_charco_ids_projector, board, projector_shape, None, None, flags=cv2.CALIB_FIX_K2+cv2.CALIB_FIX_K3+cv2.CALIB_FIX_K4+cv2.CALIB_FIX_K5+cv2.CALIB_FIX_K6)

#np.savez("../camera_calibration_out/calculated_cams_matrix.npz", rep_err_camera=rep_err_camera, mtx_camera=mtx_camera, dist_camera=dist_camera , rvecs_camera=rvecs_camera, tvecs_camera=tvecs_camera, newcameramtx_camera=newcameramtx_camera, roi_camera=newcameramtx_camera,
 #        rep_err_proj=rep_err_proj, mtx_proj=mtx_proj, dist_proj=dist_proj, rvecs_proj=rvecs_proj, tvecs_proj=tvecs_proj, newcameramtx_proj=newcameramtx_proj, roi_proj=newcameramtx_proj)

retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
    cv2.stereoCalibrate(all_obj_points, all_charco_corners_camera_2, all_charco_corners_projector,
                        mtx_camera, dist_camera, mtx_proj,
                        dist_proj, camera_resolution, flags=cv2.CALIB_FIX_INTRINSIC)

newcameramtx_camera, roi_camera=cv2.getOptimalNewCameraMatrix(cameraMatrix1,distCoeffs1,camera_resolution,1, camera_resolution)
newcameramtx_proj, roi_proj=cv2.getOptimalNewCameraMatrix(cameraMatrix2,distCoeffs2,projector_shape,1, projector_shape)

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
rep_err_proj, mtx_proj, dist_proj, rvecs_proj, tvecs_proj = cv2.aruco.calibrateCameraCharuco(all_charco_corners_projector, all_charco_ids_projector, board, projector_shape, None, None)

retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
    cv2.stereoCalibrate(all_obj_points, all_charco_corners_camera_2, all_charco_corners_projector,
                        mtx_camera, dist_camera, mtx_proj,
                        dist_proj, camera_resolution, flags=cv2.CALIB_FIX_INTRINSIC)

# yet again, compute optimal undistorted matrices
newcameramtx_camera, roi_camera=cv2.getOptimalNewCameraMatrix(cameraMatrix1,distCoeffs1,camera_resolution,1, camera_resolution)
newcameramtx_proj, roi_proj=cv2.getOptimalNewCameraMatrix(cameraMatrix2,distCoeffs2,projector_shape,1, projector_shape)

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

print(f"Stereo calibration results:")
print(f"Stereo reprojection error: {retval}")
print(f"Rotation matrix / translation vector from camera to projector:")
print(f"Rotation matrix:\n{R}")
print(f"Translation vector:\n{T}")
