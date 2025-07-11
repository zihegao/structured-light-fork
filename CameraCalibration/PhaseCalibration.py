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

# CONFIGURATION
TOTAL_SETS = 8
PROJ_W = 1920
PROJ_H = 1080

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root  = os.path.abspath(os.path.join(script_dir, '..'))

    calib_root = os.path.join("..", 'captures', 'Calib3')
    out_folder = os.path.join("..", 'camera_calibration_out')
    os.makedirs(out_folder, exist_ok=True)

    sets = [f'c_{i}' for i in range(TOTAL_SETS)]
    valid_sets = [s for s in sets if os.path.isdir(os.path.join(calib_root, s))]
    if not valid_sets:
        print(f"ERROR: no c_# folders found under {calib_root}")
        return

    print("Found calibration sets:")
    for s in valid_sets:
        print("  ", s)

    cam_pts_list, cam_id_list = [], []
    prj_pts_list, prj_id_list = [], []
    world_pts_list = []

    for s in valid_sets:
        folder = os.path.join(calib_root, s)
        w_img = cv2.imread(os.path.join(folder, 'w.jpg'))
        if w_img is None:
            print(f"WARN: missing w.jpg → skipping {s}")
            continue

        gray = cv2.cvtColor(w_img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, BoardInfo.arucoDict)
        if ids is None or len(ids) == 0:
            print(f"WARN: no ArUco markers in {s}")
            continue

        _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            corners, ids, gray, BoardInfo.charucoBoard)
        if charuco_ids is None or len(charuco_ids) < 4:
            n = 0 if charuco_ids is None else len(charuco_ids)
            print(f"WARN: only {n} CharUco corners in {s}")
            continue

        x_unwrapped = cv2.imread(os.path.join(folder, 'x_unwrapped.tiff'), cv2.IMREAD_UNCHANGED)
        y_unwrapped = cv2.imread(os.path.join(folder, 'y_unwrapped.tiff'), cv2.IMREAD_UNCHANGED)
        invH = cv2.imread(os.path.join(folder, 'out_InvalidImageH.tiff'), cv2.IMREAD_GRAYSCALE)
        invV = cv2.imread(os.path.join(folder, 'out_InvalidImageV.tiff'), cv2.IMREAD_GRAYSCALE)
        if any(v is None for v in (x_unwrapped, y_unwrapped, invH, invV)):
            print(f"WARN: missing unwrapped or mask files in {s}")
            continue

        valid, cam_pts, prj_pts = getCameraCoordinates(
            w_img, invV, invH, x_unwrapped, y_unwrapped, charuco_corners
        )
        if cam_pts.shape[0] < 4:
            print(f"WARN: only {cam_pts.shape[0]} valid correspondences in {s}")
            continue

        print(f"{s}: collected {cam_pts.shape[0]} correspondences")

        cam_pts_list.append(cam_pts.reshape(-1,1,2).astype(np.float32))
        cam_id_list.append(charuco_ids.reshape(-1,1).astype(np.int32))
        prj_pts_list.append(prj_pts.reshape(-1,1,2).astype(np.float32))
        prj_id_list.append(charuco_ids.reshape(-1,1).astype(np.int32))
        world_pts_list.append(BoardInfo.charucoBoard.getChessboardCorners().astype(np.float32))

    if len(cam_pts_list) < 2:
        print(f"ERROR: need ≥2 valid sets, found {len(cam_pts_list)}")
        return

    h, w = gray.shape
    rep_err_cam, camM1, dist1, _, _ = aruco.calibrateCameraCharuco(
        cam_pts_list, cam_id_list, BoardInfo.charucoBoard, (w, h), None, None)
    print(f"\nCamera RMS reproj err: {rep_err_cam:.2f} px")

    rep_err_prj, camM2, dist2, _, _ = aruco.calibrateCameraCharuco(
        prj_pts_list, prj_id_list, BoardInfo.charucoBoard, (PROJ_W, PROJ_H), None, None)
    print(f"Projector RMS reproj err: {rep_err_prj:.2f} px")

    flags = cv2.CALIB_FIX_INTRINSIC
    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        world_pts_list, cam_pts_list, prj_pts_list,
        camM1, dist1, camM2, dist2, (w, h), flags=flags)
    print(f"Stereo RMS reproj err: {ret:.2f} px")

    np.savez(os.path.join(out_folder, 'calculated_cams_matrix.npz'),
             retval=ret,
             cameraMatrix1=camM1, distCoeffs1=dist1,
             cameraMatrix2=camM2, distCoeffs2=dist2,
             R=R, T=T, E=E, F=F)
    print("\nSaved → calculated_cams_matrix.npz")

if __name__ == '__main__':
    main()
