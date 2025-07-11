#!/usr/bin/env python3
"""
BL.py

Load your stereo‐calibration NPZ (calculated_cams_matrix.npz),
then extract and print:

  • Baseline B  = ‖T‖  (the camera–projector translation norm, in the same units as your Charuco board)
  • Projector focal length L  = fx_proj = cameraMatrix2[0,0] (in pixels)
  • Camera focal length (for reference)

Usage:
    python BL.py --calib path/to/calculated_cams_matrix.npz
"""

import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser(
        description="Extract Baseline (B) and focal lengths from stereo‐calibration NPZ"
    )
    parser.add_argument(
        "--calib",
        type=str,
        default="./camera_calibration_out/calculated_cams_matrix.npz",
        help="Path to your calculated_cams_matrix.npz"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.calib):
        print(f"ERROR: Cannot find calibration file at '{args.calib}'")
        return

    C = np.load(args.calib)

    # T is stored as a (3×1) or (3,) array
    T = C["T"].reshape(-1)
    B = np.linalg.norm(T)
    print(f"Baseline (B)                     : {B:.6f}  (in same units as your Charuco board)")

    # Projector intrinsics
    if "cameraMatrix2" in C:
        mtx_proj = C["cameraMatrix2"]
        fx_proj = mtx_proj[0, 0]
        fy_proj = mtx_proj[1, 1]
        cx_proj = mtx_proj[0, 2]
        cy_proj = mtx_proj[1, 2]
        print(f"Projector focal length (fx, fy)  : ({fx_proj:.3f} px, {fy_proj:.3f} px)")
        print(f"Projector principal point (cx,cy): ({cx_proj:.3f}, {cy_proj:.3f})  (in px)")
    else:
        print("WARNING: 'cameraMatrix2' not found in NPZ; cannot print projector intrinsics.")

    # Camera intrinsics (for reference)
    if "cameraMatrix1" in C:
        mtx_cam = C["cameraMatrix1"]
        fx_cam = mtx_cam[0, 0]
        fy_cam = mtx_cam[1, 1]
        cx_cam = mtx_cam[0, 2]
        cy_cam = mtx_cam[1, 2]
        print(f"Camera focal length (fx, fy)     : ({fx_cam:.3f} px, {fy_cam:.3f} px)")
        print(f"Camera principal point (cx,cy)   : ({cx_cam:.3f}, {cy_cam:.3f})  (in px)")
    else:
        print("WARNING: 'cameraMatrix1' not found in NPZ; cannot print camera intrinsics.")

    # Just to be thorough, show the translation vector
    print(f"Translation vector T             : [{T[0]:.6f}, {T[1]:.6f}, {T[2]:.6f}]")

if __name__ == "__main__":
    main()
