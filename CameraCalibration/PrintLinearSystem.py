# =============================================================================
# File:  PrintLinearSystem.py
#
# used as a debug information file that will print the important math relating to this 
# =============================================================================

import os
import cv2
import numpy as np
import sympy as sp
from cv2 import aruco

# ensure this file is in Camera Calibration to work 
import BoardInfo
from GetSecondViewPoints import getCameraCoordinates


# user input hard data (currently using main captures folder)
DEFAULT_FOLDER  = "./captures/Calib3/c_0"
DEFAULT_PROJ_W  = 1920
DEFAULT_PROJ_H  = 1080
# ──────────────────────────────────────────────────────────────────────────


def collect_correspondences_gray(calib_folder, proj_w, proj_h):
    """
    For a Gray-Code calibration folder, collect:
      1) Camera 2D Charuco corners: (u_i_cam, v_i_cam)
      2) Board 3D corners:          (X_i,  Y_i,  Z_i)
      3) Projector 2D points:       (u_i_proj, v_i_proj)
         from out_BinImageH.tiff / out_BinImageV.tiff
         and out_InvalidImageH/V.tiff
    Returns (board_pts (n×3), cam_pts2D (n×2), proj_pts2D (n×2)).
    """

    # 1) load w
    w_path = os.path.join(calib_folder, "w.jpg")
    if not os.path.isfile(w_path):
        raise FileNotFoundError(f"Cannot find '{w_path}'")

    img_color = cv2.imread(w_path)
    if img_color is None:
        raise RuntimeError(f"Failed to read '{w_path}' as an image.")
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # 2) detect corners
    corners, ids, _ = aruco.detectMarkers(gray, BoardInfo.arucoDict)
    if ids is None or len(ids) == 0:
        raise RuntimeError(f"No ArUco markers found in '{w_path}'")

    _, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(
        corners, ids, gray, BoardInfo.charucoBoard
    )
    if charucoIds is None or len(charucoIds) == 0:
        raise RuntimeError(f"No Charuco corners interpolated in '{w_path}'")


    # flatten IDs for indexing:
    charucoIds = charucoIds.flatten()  # shape (N,)

    # 3) world (X,Y,Z) for each detected Charuco ID:
    all_board_points = BoardInfo.charucoBoard.getChessboardCorners()  # (totalCorners,3)
    board_pts_list = []
    for cid in charucoIds:
        board_pts_list.append(tuple(all_board_points[cid, :].tolist()))
    board_pts = np.array(board_pts_list, dtype=np.float64)  # shape (N,3)

    # 4) load decoded grey tiff files
    x_bin_path = os.path.join(calib_folder, "out_BinImageH.tiff")
    y_bin_path = os.path.join(calib_folder, "out_BinImageV.tiff")
    invH_path  = os.path.join(calib_folder, "out_InvalidImageH.tiff")
    invV_path  = os.path.join(calib_folder, "out_InvalidImageV.tiff")

    x_index = cv2.imread(x_bin_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    y_index = cv2.imread(y_bin_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    invH    = cv2.imread(invH_path,  cv2.IMREAD_GRAYSCALE)
    invV    = cv2.imread(invV_path,  cv2.IMREAD_GRAYSCALE)

    if any(img is None for img in (x_index, y_index, invH, invV)):
        raise RuntimeError(f"Missing or unreadable Gray-Code decode files in '{calib_folder}'")

    # 5) filter & match: getCameraCoordinates handles invalid-mask filtering.
    #    pass charucoCorners as a (N,1,2) array—do NOT flatten it!
    valid_mask, cam_pts2D_filt, proj_pts2D = getCameraCoordinates(
        img_color,
        invV,          # invalid mask for vertical
        invH,          # invalid mask for horizontal
        y_index,       # coordsV_map (vertical index)
        x_index,       # coordsH_map (horizontal index)
        charucoCorners # shape (N,1,2)
    )

    # valid_mask: boolean array of length N
    # cam_pts2D_filt and proj_pts2D may come back with an extra singleton dimension.


    cam_pts2D_arr  = np.array(cam_pts2D_filt)
    proj_pts2D_arr = np.array(proj_pts2D)

    # Drop any singleton dims so each becomes shape (m,2)
    cam_pts2D_final = np.squeeze(cam_pts2D_arr)
    proj_pts2D_final = np.squeeze(proj_pts2D_arr)

    # if after squeeze it’s still not 2D throw error:
    if cam_pts2D_final.ndim != 2 or cam_pts2D_final.shape[1] != 2:
        raise RuntimeError(f"cam_pts2D_filt has unexpected shape {cam_pts2D_arr.shape}")
    if proj_pts2D_final.ndim != 2 or proj_pts2D_final.shape[1] != 2:
        raise RuntimeError(f"proj_pts2D has unexpected shape {proj_pts2D_arr.shape}")
    

    # keep valid corner rows
    board_pts       = board_pts[valid_mask, :]
    cam_pts2D_final = cam_pts2D_final[valid_mask, :]
    proj_pts2D_final = proj_pts2D_final[valid_mask, :]

    if board_pts.shape[0] < 4:
        raise RuntimeError(f"Only {board_pts.shape[0]} valid corners in '{calib_folder}'. Need ≥4.")

    return board_pts, cam_pts2D_final, proj_pts2D_final


def build_symbolic_A(n):
    """
    Build a symbolic 2n×12 matrix A (for n correspondences).
    Returns (A_sym, X_syms, Y_syms, Z_syms, U_syms, V_syms).
    """
    X_syms = sp.symbols(f'x0:{n}', real=True)
    Y_syms = sp.symbols(f'y0:{n}', real=True)
    Z_syms = sp.symbols(f'z0:{n}', real=True)
    U_syms = sp.symbols(f'u0:{n}', real=True)
    V_syms = sp.symbols(f'v0:{n}', real=True)

    A_sym = sp.zeros(2 * n, 12)
    for i in range(n):
        xi, yi, zi = X_syms[i], Y_syms[i], Z_syms[i]
        ui, vi     = U_syms[i], V_syms[i]

        # Build two 1×12 rows:
        row1 = [xi, yi, zi, 1, 0, 0, 0, 0, -ui*xi, -ui*yi, -ui*zi, -ui]
        row2 = [0, 0, 0, 0, xi, yi, zi, 1, -vi*xi, -vi*yi, -vi*zi, -vi]


        A_sym[2*i,   :] = sp.Matrix([row1])  # shape (1,12)
        A_sym[2*i+1, :] = sp.Matrix([row2])  # shape (1,12)
        # ───────────────────────────────────────────────────────────────────

    return A_sym, X_syms, Y_syms, Z_syms, U_syms, V_syms


def substitute_numeric(A_sym, X_syms, Y_syms, Z_syms, U_syms, V_syms,
                       board_pts, image_uv):
    """
    Given:
      • A_sym    : 2n×12 symbolic
      • X_syms.. : symbolic lists of length n
      • board_pts: (n×3) numpy
      • image_uv: (n×2) numpy
    Build subs-dict and compute A_num = A_sym.subs(subs_dict).
    """
    n = board_pts.shape[0]
    subs = {}
    for i in range(n):
        subs[X_syms[i]] = float(board_pts[i, 0])
        subs[Y_syms[i]] = float(board_pts[i, 1])
        subs[Z_syms[i]] = float(board_pts[i, 2])
        subs[U_syms[i]] = float(image_uv[i, 0])  # now image_uv is shape (n,2)
        subs[V_syms[i]] = float(image_uv[i, 1])
    A_num = A_sym.subs(subs)
    return A_num


def main():
    # use hard-coded defaults:
    calib_folder = DEFAULT_FOLDER
    proj_w, proj_h = DEFAULT_PROJ_W, DEFAULT_PROJ_H

    if not os.path.isdir(calib_folder):
        print(f"ERROR: '{calib_folder}' is not a directory.")
        return

    board_pts, cam_uv, proj_uv = collect_correspondences_gray(
        calib_folder, proj_w, proj_h
    )
    n = board_pts.shape[0]
    print(f"\nCollected {n} valid Gray-Code correspondences from '{calib_folder}'.\n")

    # build the symbolic A-matrix
    A_sym, X_syms, Y_syms, Z_syms, U_syms, V_syms = build_symbolic_A(n)

    # sub for camera
    A_cam_num = substitute_numeric(
        A_sym, X_syms, Y_syms, Z_syms, U_syms, V_syms,
        board_pts, cam_uv
    )
    print("=== Camera A_matrix (2n×12) for P_cam: ===")
    sp.pprint(A_cam_num)

    # sub for the projector
    A_proj_num = substitute_numeric(
        A_sym, X_syms, Y_syms, Z_syms, U_syms, V_syms,
        board_pts, proj_uv
    )
    print("\n\n=== Projector A_matrix (2n×12) for P_proj: ===")
    sp.pprint(A_proj_num)

    # If you ever want purely symbolic form, uncomment:
    # print("\n\n=== Symbolic A (2n×12) w/o substitution: ===")
    # sp.pprint(A_sym)


if __name__ == "__main__":
    main()
