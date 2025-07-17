# Camera Calibration using ChArUco Boards and Gray Code (or any) Structured Light
# Detects ChArUco corners in images, maps them to projector coordinates using Gray Code (or any structured light) decoding,
# and performs stereo calibration between a camera and a projector. Outputs include intrinsic and extrinsic matrices.

# Forked from https://github.com/sjnarmstrong/gray-code-structured-light with code also from https://github.com/kamino410/procam-calibration

import os
import glob
import cv2
from calibration_board import board
import numpy as np
from cv2 import aruco

PATCH_SIZE = 47 # local homography patch size

def printNumpyWithIndent(tar, indentchar):
    print(indentchar + str(tar).replace('\n', '\n' + indentchar))

def calibrate(data_dirs, 
              board, cam_shape, proj_shape, output_dir):
    """
    Function to calibrate camera and projector using ChArUco boards and decoded structured-light correspondence.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print('Calibrating ...')

    # the following lists store calibration points for all board configrurations (frames)
    # each item in the list is a Nx1x2 array of 2D points (N is the number of detected corners in that configuration/frame)
    # len(list) = number of configurations/frames
    cam_corners_list = [] # charuco corners in camera coordinates
    cam_corners_jointly_valid_list = [] # if charuco corners have valid projector coordinates (corner always has valid camera coords)
    charuco_ids_list = [] # charuco IDs
    charuco_ids_jointly_valid_list = [] # if charuco IDs have valid projector coordinates (corner always has valid camera coords)
    proj_corners_list = [] # charuco corners in projector coordinates
    is_corner_jointly_valid_list = [] # if corner has valid projector coordinates (corner always has valid camera coords)
    obj_points_list = [] # all known object points on the calibration board that have both valid camera and projector coordinates

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
        valid_v = cv2.imread(os.path.join(dirname, "out_InvalidImageV.tiff"), cv2.IMREAD_GRAYSCALE)
        valid_h = cv2.imread(os.path.join(dirname, "out_InvalidImageH.tiff"), cv2.IMREAD_GRAYSCALE)
        coords_v = cv2.imread(os.path.join(dirname, "out_BinImageV.tiff"), cv2.IMREAD_ANYDEPTH+cv2.IMREAD_GRAYSCALE)
        coords_h = cv2.imread(os.path.join(dirname, "out_BinImageH.tiff"), cv2.IMREAD_ANYDEPTH+cv2.IMREAD_GRAYSCALE)

        ######### Aruco marker and Charuco corner detection #########
        # detect Aruco markers
        aruco_corners, ids, rejected = aruco.detectMarkers(img, board.getDictionary())

        # try to recover Aruco markers based on board info if they are not directly detected
        if len(aruco_corners) > 0:
            aruco_corners, ids, rejected, recovered = aruco.refineDetectedMarkers(
                img, board, aruco_corners, ids, rejected)

        # optionally visualize detected Aruco markers
        cimg = aruco.drawDetectedMarkers(img.copy(), aruco_corners, ids)
        cv2.imwrite(outPath+"/DetectedMarkers.png", cimg)

        # interpolate ChArUco corners using ArUco markers and board info
        numCorners, charucoCorners, charuco_ids = aruco.interpolateCornersCharuco(aruco_corners, ids, img, board)

        if numCorners < 4:  # need at least 4 corners to compute homography
            print(f'    Warning: only {numCorners} corners detected in {dirname}, skipping this set')
            continue
    
        cam_corners_list.append(charucoCorners)
        charuco_ids_list.append(charuco_ids)
        
        #### get projector coordinates for each detected corner using local homography ####

        is_corner_valid, proj_corners = get_projector_corner_coords(img, valid_v, valid_h, coords_v, coords_h, charucoCorners, patch_size=PATCH_SIZE)
        
        if len(proj_corners) < 4:
            print(f'    Warning: not enough corners with valid projector coordinates in {dirname}, skipping this frame')
            continue
        proj_corners_list.append(proj_corners)
        is_corner_jointly_valid_list.append(is_corner_valid)
        cam_corners_jointly_valid_list.append(charucoCorners[is_corner_valid])
        charuco_ids_jointly_valid_list.append(charuco_ids[is_corner_valid])

        # get known object points on the board that are detected, save to list for stereo calibration
        # obj_points_list.append(board.getChessboardCorners()[charuco_ids.flatten()[is_corner_valid], :]) # this gives 48x3 array
        obj_points_list.append(board.getChessboardCorners()[charuco_ids[is_corner_valid], :]) # this gives 48x1x3 array

    ###### Perform Calibration ######
    cam_reproj_err, cam_mat, cam_dist, cam_rvecs, cam_tvecs = cv2.aruco.calibrateCameraCharuco(
        cam_corners_list, charuco_ids_list, board, cam_shape, None, None, 
        # flags=cv2.CALIB_FIX_K2+cv2.CALIB_FIX_K3+cv2.CALIB_FIX_K4+cv2.CALIB_FIX_K5+cv2.CALIB_FIX_K6
        )
    proj_reproj_err, proj_mat, proj_dist, proj_rvecs, proj_tvecs = cv2.aruco.calibrateCameraCharuco(
            proj_corners_list, charuco_ids_jointly_valid_list, board, proj_shape, None, None,
            # flags=cv2.CALIB_FIX_K2+cv2.CALIB_FIX_K3+cv2.CALIB_FIX_K4+cv2.CALIB_FIX_K5+cv2.CALIB_FIX_K
            )
    stereo_reproj_err, cam_mat2, cam_dist2, proj_mat2, proj_dist2, R, T, E, F = cv2.stereoCalibrate(
        obj_points_list, cam_corners_jointly_valid_list, proj_corners_list,
        cam_mat, cam_dist, proj_mat, proj_dist, 
        None, None,
        flags=cv2.CALIB_FIX_INTRINSIC
        )
    
    print(' Camera calibration results')
    print('  Reprojection error :', cam_reproj_err)
    print('  Intrinsic parameters :')
    printNumpyWithIndent(cam_mat, '    ')
    print('  Distortion parameters :')
    printNumpyWithIndent(cam_dist, '    ')
    print()

    print('Initial solution of projector\'s parameters')
    print('  Reprojection error :', proj_reproj_err)
    print('  Intrinsic parameters :')
    printNumpyWithIndent(proj_mat, '    ')
    print('  Distortion parameters :')
    printNumpyWithIndent(proj_dist, '    ')
    print()

    print('=== Joint calibration result ===')
    print('  Reprojection error :', stereo_reproj_err)
    print('  Camera intrinsic parameters :')
    printNumpyWithIndent(cam_mat2, '    ')
    print('  Camera distortion parameters :')
    printNumpyWithIndent(cam_dist2, '    ')
    print('  Projector intrinsic parameters :')
    printNumpyWithIndent(proj_mat2, '    ')
    print('  Projector distortion parameters :')
    printNumpyWithIndent(proj_dist2, '    ')
    print('  Rotation matrix / translation vector from camera to projector')
    print('  (they translate points from camera coord to projector coord) :')
    printNumpyWithIndent(R, '    ')
    printNumpyWithIndent(T, '    ')
    print()
    

    return cam_reproj_err, cam_mat2, cam_dist2, proj_reproj_err, proj_mat2, proj_dist2, stereo_reproj_err, R, T, E, F
            # newcameramtx_camera, roi_camera=cv2.getOptimalNewCameraMatrix(cam_mat2,cam_dist2,camera_resolution,1, camera_resolution)
            # newcameramtx_proj, roi_proj=cv2.getOptimalNewCameraMatrix(proj_mat2,proj_dist2,projector_shape,1, projector_shape)

    # invCamMtx = np.linalg.inv(newcameramtx_camera)
    # invProjMtx = np.linalg.inv(newcameramtx_proj)

# save less distorted calibration results
    # np.savez(output_dir+"/calculated_cams_matrix_less_distortion.npz",
    #         retval=stereo_reproj_err,
    #         cameraMatrix1=cam_mat2,
    #         distCoeffs1=cam_dist2,
    #         cameraMatrix2=proj_mat2,
    #         distCoeffs2=proj_dist2,
    #         R=R,
    #         T=T,
    #         E=E,
    #         F=F,
    #         newcameramtx_camera=newcameramtx_camera,
    #         roi_camera=roi_camera,
    #         newcameramtx_proj=newcameramtx_proj,
    #         roi_proj=roi_proj,
    #         invCamMtx=invCamMtx,
    #         invProjMtx=invProjMtx)


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
        patch_coords_valid = patch_idx[:, [1, 0]]

        # get corresponding projector coordinates for all valid points in patch
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
            patch_coords_valid,  
            proj_coords, 
            ransacReprojThreshold=2, maxIters=100000, method = cv2.FM_LMEDS, confidence=0.99
            )
        if H is None:
            is_corner_valid.append(False)
            print(f'    Warning: corner {int(corner[0,0]), int(corner[0,1])} \
                  was skipped because homography computation failed (check your images and thresholds)')
            continue

        # map the current corner point through the homography to get projector coordinates
        proj_corner = np.dot(H, [corner[0,0], corner[0,1], 1.0])
        proj_corner /= proj_corner[2] # homogeneous coordinate normalization
        proj_corners.append([[proj_corner[0], proj_corner[1]]])
        is_corner_valid.append(True)

    return is_corner_valid, np.array(proj_corners, dtype=np.float32)

if __name__ == "__main__":
    # path to calibration data
    # data_dir = """./captures/Calib3/"""   
    data_dir = """C:/Users/zihegao/Box/apl_shared/undergraduate_researchers/Ethan_Wilson/Pictures/NotWorking/Calib3/"""
    output_dir = """./camera_calibration_out/"""
    imgfmt = ".jpg"
    projector_shape =(1920, 1080)

    dirnames = sorted(glob.glob(data_dir+"c_*"))

    # check if folders constain useful files
    img_files_lists = [] # a list of lists, each containing image files for a specific calibration data folder
    dir_to_use = []
    for dirname in dirnames:
        img_files = sorted((glob.glob(os.path.join(dirname, 'w'+imgfmt))))
        if len(img_files) == 0:
            continue
        img_files_lists.append(img_files)
        dir_to_use.append(dirname)
    cam_shape = cv2.imread(img_files_lists[0][0], cv2.IMREAD_GRAYSCALE).shape
    cam_reproj_err, cam_mat2, cam_dist2, proj_reproj_err, proj_mat2, proj_dist2, stereo_reproj_err, R, T, E, F = calibrate(
        dir_to_use, board, cam_shape, projector_shape, output_dir)