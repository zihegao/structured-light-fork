# PhaseCalibration.py
# Stereo-calibrate a camera and projector using phase-shift index maps.
# Debug-heavy version: collects Charuco and phase correspondences, then reports
# intrinsic and stereo calibration (with variants) and iteratively shows T growth.

import os
import cv2
import numpy as np
from cv2 import aruco

import BoardInfo
from GetSecondViewPoints import getCameraCoordinates

# CONFIGURATION
TOTAL_SETS = 8        # Number of c_0…c_{TOTAL_SETS-1}
PROJ_W     = 1920     # Projector width
PROJ_H     = 1080     # Projector height


def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root  = os.path.abspath(os.path.join(script_dir, '..'))
    calib_root = os.path.join(repo_root, 'captures', 'Calib3')
    out_folder = os.path.join(repo_root, 'camera_calibration_out')
    os.makedirs(out_folder, exist_ok=True)

    # Gather valid sets
    sets = [f'c_{i}' for i in range(TOTAL_SETS)]
    valid_sets = [s for s in sets if os.path.isdir(os.path.join(calib_root, s))]
    if len(valid_sets) < 2:
        print(f"ERROR: need >=2 valid sets, found {len(valid_sets)}")
        return
    print("Found sets:", valid_sets)

    # Prepare lists
    cam_pts_list, cam_id_list = [], []
    prj_pts_list, prj_id_list = [], []
    world_pts_list = []
    full_corners3d = BoardInfo.charucoBoard.getChessboardCorners().astype(np.float32)

    # Step 1: collect correspondences
    for s in valid_sets:
        folder = os.path.join(calib_root, s)
        img = cv2.imread(os.path.join(folder,'w.jpg'))
        if img is None:
            print(f"WARN {s}: missing w.jpg")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, BoardInfo.arucoDict)
        _, cc, ci = aruco.interpolateCornersCharuco(corners, ids, gray, BoardInfo.charucoBoard)
        if ci is None or len(ci) < 4:
            print(f"WARN {s}: only {0 if ci is None else len(ci)} CharUco corners")
            continue

        x_map = cv2.imread(os.path.join(folder,'x_unwrapped.tiff'), cv2.IMREAD_UNCHANGED)
        y_map = cv2.imread(os.path.join(folder,'y_unwrapped.tiff'), cv2.IMREAD_UNCHANGED)
        invH  = cv2.imread(os.path.join(folder,'out_InvalidImageH.tiff'), cv2.IMREAD_GRAYSCALE)
        invV  = cv2.imread(os.path.join(folder,'out_InvalidImageV.tiff'), cv2.IMREAD_GRAYSCALE)
        try:
            mask, cpts, ppts = getCameraCoordinates(img, invV, invH, x_map, y_map, cc)
        except Exception as e:
            print(f"WARN {s}: getCameraCoordinates failed: {e}")
            continue
        n = cpts.shape[0]
        if n < 4:
            print(f"WARN {s}: only {n} valid correspondences")
            continue
        print(f"{s}: collected {n} correspondences")

        # subset world points by mask
        ids_flat = ci.flatten()
        objp_all = full_corners3d[ids_flat]
        objp = objp_all[mask]

        cam_pts_list.append(cpts.reshape(-1,1,2).astype(np.float32))
        cam_id_list.append(ci.reshape(-1,1).astype(np.int32))
        prj_pts_list.append(ppts.reshape(-1,1,2).astype(np.float32))
        prj_id_list.append(ci.reshape(-1,1).astype(np.int32))
        world_pts_list.append(objp)

    # Step 2: intrinsic calibrations
    sample = cv2.imread(os.path.join(calib_root, valid_sets[0],'w.jpg'))
    h, w = cv2.cvtColor(sample,cv2.COLOR_BGR2GRAY).shape
    err_cam, camM1, dist1, _, _ = aruco.calibrateCameraCharuco(
        cam_pts_list, cam_id_list, BoardInfo.charucoBoard, (w,h), None, None)
    err_prj, camM2, dist2, _, _ = aruco.calibrateCameraCharuco(
        prj_pts_list, prj_id_list, BoardInfo.charucoBoard, (PROJ_W,PROJ_H), None, None)
    print(f"Camera RMS err: {err_cam:.2f}px, Projector RMS err: {err_prj:.2f}px")

    # Step 3: stereoCalibrate variants
    flags = cv2.CALIB_FIX_INTRINSIC
    print("\n[DEBUG] stereoCalibrate FIX_INTRINSIC:")
    ret1, _, _, _, _, R1, T1, E1, F1 = cv2.stereoCalibrate(
        world_pts_list, cam_pts_list, prj_pts_list,
        camM1, dist1, camM2, dist2, (w,h), flags=flags)
    print(f"err={ret1:.2f}, |T|={np.linalg.norm(T1):.2f}, T={T1.ravel()}")

    print("[DEBUG] stereoCalibrate no flags:")
    ret2, _, _, _, _, R2, T2, E2, F2 = cv2.stereoCalibrate(
        world_pts_list, cam_pts_list, prj_pts_list,
        camM1, dist1, camM2, dist2, (w,h), flags=0)
    print(f"err={ret2:.2f}, |T|={np.linalg.norm(T2):.2f}, T={T2.ravel()}")

    print("[DEBUG] stereoCalibrate USE_INTRINSIC_GUESS:")
    ret3, _, _, _, _, R3, T3, E3, F3 = cv2.stereoCalibrate(
        world_pts_list, cam_pts_list, prj_pts_list,
        camM1, dist1, camM2, dist2, (w,h), flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    print(f"err={ret3:.2f}, |T|={np.linalg.norm(T3):.2f}, T={T3.ravel()}")

    # Step 4: iterative T growth
    print("\n[DEBUG] iterative T (1..k):")
    for k in range(2, len(world_pts_list)+1):
        ret_k, _, _, _, _, Rk, Tk, Ek, Fk = cv2.stereoCalibrate(
            world_pts_list[:k], cam_pts_list[:k], prj_pts_list[:k],
            camM1, dist1, camM2, dist2, (w,h), flags=flags)
        print(f"1..{k}: err={ret_k:.2f}, |T|={np.linalg.norm(Tk):.2f}")

    # Final summary
    rvec, _ = cv2.Rodrigues(R1)
    print("\n==== Final Summary ====")
    print(f"Rotation: {rvec.ravel()}")
    print(f"Translation T: {T1.ravel()} |T|={np.linalg.norm(T1):.2f}")
    print(f"Cam Intrinsics: {camM1.flatten()}")
    print(f"Prj Intrinsics: {camM2.flatten()}")

    # Save debug calibration
    np.savez(os.path.join(out_folder, 'calibration_debug.npz'),
             retval=ret1,
             cameraMatrix1=camM1, distCoeffs1=dist1,
             cameraMatrix2=camM2, distCoeffs2=dist2,
             R=R1, T=T1, E=E1, F=F1)
    print("Saved → calibration_debug.npz")

if __name__ == '__main__':
    main()
