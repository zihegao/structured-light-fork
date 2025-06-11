#
# PhaseCalibration.py
# Stereo-calibrate a camera and projector using phase-shift index maps.
# Configuration at top of file:
# TOTAL_SETS : Number of c_<i> folders to process under 'captures/Calib3'
# Each c_<i> folder must contain:
#  w.jpg
#  x_index.tiff
#  y_index.tiff
#  out_InvalidImageH.tiff
#  out_InvalidImageV.tiff
#
#  To run:
#  python PhaseCalibration.py
# 
#  Outputs:
#  captures/Calib3/camera_calibration_out/calculated_cams_matrix.npz

import os
import cv2
import numpy as np
from cv2 import aruco

import BoardInfo
from GetSecondViewPoints import getCameraCoordinates

# ───────────────────────────────────────────────────
#  USER CONFIGURATION 
TOTAL_SETS = 7  # adjust to number of calibration folders c_0...c_{TOTAL_SETS-1} # ATTENTION FIXME

# ───────────────────────────────────────────
#  PROJECTOR RESOLUTION (stable) 
PROJ_W = 1920  # projector width in pixels
PROJ_H = 1080  # projector height in pixels
# ──────────────────────────────────────────────────────────────────────────

def main():
    # determine paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, '..'))
    calib_root = os.path.join(repo_root, 'captures', 'Calib3')
    out_folder = os.path.join(calib_root, 'camera_calibration_out')

    if not os.path.isdir(calib_root):
        print(f"ERROR: Calibration directory not found: {calib_root}")
        return
    os.makedirs(out_folder, exist_ok=True)

    # collect the valid set folders
    set_folders = [f'c_{i}' for i in range(TOTAL_SETS)]
    valid_folders = [f for f in set_folders if os.path.isdir(os.path.join(calib_root, f))]
    if not valid_folders:
        print(f"ERROR: No c_<i> folders found in {calib_root}")
        return

    print("Using calibration sets:")
    for f in valid_folders:
        print("  ", f)

    cam_corners = []
    cam_ids     = []
    proj_corners = []
    proj_ids     = []
    world_pts   = []

    # iterate through the calibration sets c_0 to c_# sets
    for folder in valid_folders:
        folder_path = os.path.join(calib_root, folder)
        w_path = os.path.join(folder_path, 'w.jpg')
        img = cv2.imread(w_path)
        if img is None:
            print(f"WARN: Missing w.jpg in {folder}, skipping.")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # detects the aruco markers
        corners, ids, _ = aruco.detectMarkers(gray, BoardInfo.arucoDict)
        if ids is None or len(ids) == 0:
            print(f"WARN: No ArUco markers in {folder}, skipping.")
            continue
        _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, gray, BoardInfo.charucoBoard)
        if charuco_ids is None or len(charuco_ids) < 4:
            print(f"WARN: Only {0 if charuco_ids is None else len(charuco_ids)} corners in {folder}, skipping.")
            continue

        # load phase decode maps
        x_idx = cv2.imread(os.path.join(folder_path, 'x_index.tiff'), cv2.IMREAD_ANYDEPTH)
        y_idx = cv2.imread(os.path.join(folder_path, 'y_index.tiff'), cv2.IMREAD_ANYDEPTH)
        invH  = cv2.imread(os.path.join(folder_path, 'out_InvalidImageH.tiff'), cv2.IMREAD_GRAYSCALE)
        invV  = cv2.imread(os.path.join(folder_path, 'out_InvalidImageV.tiff'), cv2.IMREAD_GRAYSCALE)
        if x_idx is None or y_idx is None or invH is None or invV is None:
            print(f"WARN: Missing decode outputs in {folder}, skipping.")
            continue

        # filter correspondences
        valid_mask, cam_pts, proj_pts = getCameraCoordinates(
            img, invV, invH, y_idx, x_idx, charuco_corners)
        if cam_pts.shape[0] < 4:
            print(f"WARN: Only {cam_pts.shape[0]} valid points in {folder}, skipping.")
            continue

        # save
        cam_corners.append(cam_pts.reshape(-1,1,2).astype(np.float32))
        cam_ids.append(charuco_ids.reshape(-1,1).astype(np.int32))
        proj_corners.append(proj_pts.reshape(-1,1,2).astype(np.float32))
        proj_ids.append(charuco_ids.reshape(-1,1).astype(np.int32))
        world_pts.append(BoardInfo.charucoBoard.getChessboardCorners().astype(np.float32))
        print(f"Collected {cam_pts.shape[0]} points from {folder}.")

    if len(cam_corners) < 2:
        print(f"ERROR: Need ≥2 valid sets, found {len(cam_corners)}.")
        return

    # calibrate intrinsics
    img_h, img_w = gray.shape
    cam_res = (img_w, img_h)
    proj_res = (PROJ_W, PROJ_H)
    _, camM1, dist1, _, _ = cv2.aruco.calibrateCameraCharuco(cam_corners, cam_ids, BoardInfo.charucoBoard, cam_res, None, None)
    _, camM2, dist2, _, _ = cv2.aruco.calibrateCameraCharuco(proj_corners, proj_ids, BoardInfo.charucoBoard, proj_res, None, None)

    # stereo calibrate
    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        world_pts, cam_corners, proj_corners,
        camM1, dist1, camM2, dist2, cam_res,
        flags=cv2.CALIB_FIX_INTRINSIC)
    print(f"Stereo reproj error: {ret:.4f}")

    # save
    np.savez(os.path.join(out_folder, 'calculated_cams_matrix.npz'),
             retval=ret, cameraMatrix1=camM1, distCoeffs1=dist1,
             cameraMatrix2=camM2, distCoeffs2=dist2,
             R=R, T=T, E=E, F=F)
    print(f"Saved calibration to {out_folder}")

if __name__ == '__main__':
    main()