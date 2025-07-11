#!/usr/bin/env python3
"""
IDSCaptureCode.py

Structured light capture using IDS camera, supporting both phase-shifting and Gray-code modes.
Toggle mode via PHASE_MODE: True for phase-shifting, False for Gray-code.
Captures saved under master/captures/Calib3 or master/captures/<object_name>.

Enhanced: if calibration corners aren't fully detected after white capture,
closes windows and restarts the capture process, prompting user to reposition.
"""
import os
import sys
import cv2
import numpy as np
import time
import structuredlight as sl
from interface import IDSinterface
from cv2 import aruco

# Add project root to path for BoardInfo import
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, '..'))
calib_path = os.path.join(repo_root, 'CameraCalibration')
if calib_path not in sys.path:
    sys.path.insert(0, calib_path)
import BoardInfo

# --------------------------------------------------------------------
# Configuration
WINDOW_NAME       = "StructuredLight"
PROJ_WIDTH        = 1920
PROJ_HEIGHT       = 1080
NUM_PHASE_IMAGES  = 3       # number of fringes
REF_SETTLE_MS     = 1500    # ms for white/black capture
PHASE_SETTLE_MS   = 1000    # ms for phase patterns
VERT_EXTRA_DELAY  = 1500    # extra before vertical
HORIZ_EXTRA_DELAY = 0       # extra before horizontal
PHASE_MODE        = False   # True=phase-shift, False=Gray-code
EXPOSURE_US       = 75000   # camera exposure in microseconds
GAIN              = 1.4   # camera gain

# --------------------------------------------------------------------
def wait_for_render(ms):
    end = time.time() + ms/1000.0
    while time.time() < end:
        cv2.waitKey(1)


def ids_show_and_capture(cam, img_pattern, settle_ms):
    cv2.imshow(WINDOW_NAME, img_pattern)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.waitKey(1)
    for _ in range(5): cam.capture()
    wait_for_render(settle_ms)
    cv2.waitKey(1)
    for _ in range(2): cam.capture()
    frame = cam.capture()
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 1:
        return frame[:, :, 0]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def main():
    while True:
        # Select capture type
        choice = input("Enter 'c' for calibration set or 'o' for object capture: ").strip().lower()
        if choice == 'c':
            setnum = input("Calibration set number (e.g. '3'): ").strip()
            BASE_OUTPUT_DIR = os.path.join(repo_root, 'captures', 'Calib3', f'c_{setnum}')
        elif choice == 'o':
            obj = input("Object name: ").strip()
            BASE_OUTPUT_DIR = os.path.join(repo_root, 'captures', obj)
        else:
            print("Invalid choice, exiting.")
            return
        os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

        # Initialize IDS camera
        cam = IDSinterface()
        cam.select_and_start_device()
        cam.set_gain(GAIN)
        cam.set_exposure_time(EXPOSURE_US)
        try:
            cam.set_aoi(0, 0, PROJ_WIDTH, PROJ_HEIGHT)
        except AttributeError:
            pass

        # Create fullscreen projection window
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        try:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
        except:
            pass

        # Prompt to start
        instr = np.zeros((PROJ_HEIGHT, PROJ_WIDTH), np.uint8)
        cv2.putText(instr, "Position window and press ENTER to start",
                    (50, PROJ_HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        print("Position the projection window and press ENTER to begin capture.")
        while cv2.waitKey(100) != 13:
            cv2.imshow(WINDOW_NAME, instr)

        # Warm-up render
        print("Warm-up capture (2s)...")
        dummy = np.zeros((PROJ_HEIGHT, PROJ_WIDTH), np.uint8)
        cv2.imshow(WINDOW_NAME, dummy)
        cv2.waitKey(2000)
        cam.capture(); cam.capture()

        # Capture white reference
        white_pat = np.full((PROJ_HEIGHT, PROJ_WIDTH), 255, np.uint8)
        img_w = ids_show_and_capture(cam, white_pat, REF_SETTLE_MS)
        cv2.imwrite(os.path.join(BASE_OUTPUT_DIR, 'w.jpg'), img_w)

        # If calibration, check Charuco corners once
        if choice == 'c':
            gray = img_w if img_w.ndim==2 else cv2.cvtColor(img_w, cv2.COLOR_BGR2GRAY)
            corners, ids_, _ = aruco.detectMarkers(gray, BoardInfo.arucoDict)
            _, cc, ci = aruco.interpolateCornersCharuco(corners, ids_, gray, BoardInfo.charucoBoard)
            total = BoardInfo.charucoBoard.getChessboardCorners().shape[0]
            found = 0 if ci is None else len(ci)
            print(f"Detected {found}/{total} Charuco corners.")
            if found != total:
                print("Not all corners found. Please reposition and retry capture.")
                cam.stop_acquisition()
                cv2.destroyAllWindows()
                continue   # restart from top
            else:
                print("All corners detected. Proceeding...")

        # Capture black reference
        print("Capturing black reference...")
        black_pat = np.zeros((PROJ_HEIGHT, PROJ_WIDTH), np.uint8)
        img_b = ids_show_and_capture(cam, black_pat, REF_SETTLE_MS)
        cv2.imwrite(os.path.join(BASE_OUTPUT_DIR, 'b.jpg'), img_b)

        # Capture patterns
        if PHASE_MODE:
            ps = sl.PhaseShifting(num=NUM_PHASE_IMAGES)
            print("Capturing vertical phase patterns...")
            for i, pat in enumerate(sl.transpose(ps.generate((PROJ_HEIGHT, PROJ_WIDTH)))):
                delay = PHASE_SETTLE_MS + (VERT_EXTRA_DELAY if i==0 else 0)
                img = ids_show_and_capture(cam, pat, delay)
                cv2.imwrite(os.path.join(BASE_OUTPUT_DIR, f'v{i}.jpg'), img)
            print("Capturing horizontal phase patterns...")
            for i, pat in enumerate(ps.generate((PROJ_WIDTH, PROJ_HEIGHT))):
                delay = PHASE_SETTLE_MS + (HORIZ_EXTRA_DELAY if i==0 else 0)
                img = ids_show_and_capture(cam, pat, delay)
                cv2.imwrite(os.path.join(BASE_OUTPUT_DIR, f'h{i}.jpg'), img)
        else:
            from GrayCodesWindow import getImageIteration, destroyW
            print("Capturing Gray-code patterns via GrayCodesWindow...")

            # this for loops is the important logic area
            # logic that leads to blocky invalid pixel patterns on invalid mask are generated here
            # also changed epsilom to 1 instead of 5 in the decode to help reduce count, and it did
            # can change num_avg to get different results, lower number = more invalid pixels in past cases
            # and the gf assignement at the end is where the final result goes, follow its path of origin for more

            for imgnr in getImageIteration():
                for _ in range(5): cam.capture()  # Clear camera buffer
                wait_for_render(REF_SETTLE_MS)
                for _ in range(2): cam.capture()  # Let things settle

                # Temporal average over multiple captures
                # have tried values of 3, 5, 7, and 11 which ended with the best results
                num_avg = 11  # or 9 if you want to be even smoother

                f = cam.capture()
                g = f if f.ndim == 2 else cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                acc = g.astype(np.float32)

                for _ in range(num_avg - 1):
                    f = cam.capture()
                    g = f if f.ndim == 2 else cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                    acc += g.astype(np.float32)

                avg = (acc / num_avg).astype(np.uint8)

                # Apply smoothing: Gaussian → Median → NLMeans
                blurred = cv2.GaussianBlur(avg, (5, 5), 1.2)
                column_smoothed = cv2.medianBlur(blurred, 3)
                denoised = cv2.fastNlMeansDenoising(column_smoothed, h=5, templateWindowSize=7, searchWindowSize=21)
                gf = denoised

                # Save result
                cv2.imwrite(os.path.join(BASE_OUTPUT_DIR, f'{imgnr}.jpg'), gf)
            destroyW()

        # Cleanup and finish
        cam.stop_acquisition()
        cv2.destroyAllWindows()
        print("Capture complete. Files saved in:", BASE_OUTPUT_DIR)
        break  # exit main loop

if __name__ == '__main__':
    main()
