# orchesstrator.py
# this script can run sections, or through the entire code bank process using the UI given in the terminal when run
#
# notes on capture code section
# use a notepad/white image on the projector to illuminate Charucoboard, if not all corners detected wait and try again before changing anything
# 
# when 48/48 corners found, bring popup over projector click on it, then press enter and remove mouse from screen for CaptureCode.py to run
#
# Input, is not explicit input because can start from any point in process, same with output, depends on where the user wants to start/end

# libraries for script 
import os
import sys
import cv2
import time
import shutil
import subprocess
from cv2 import aruco

# SETUP and variables to use for paths 
# ────────────────────────────────────────────────────
# this is the structure-master-light file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # = structured-light-master

CAPTURE_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "CaptureImages")

if CAPTURE_IMAGES_DIR not in sys.path:
    sys.path.insert(0, CAPTURE_IMAGES_DIR)

from GrayCodesWindow import getImageIteration, destroyW
from CaptureImage import capture_and_save_image

# ensure BoardInfo is importable
camcal_dir = os.path.join(SCRIPT_DIR, "CameraCalibration")
if camcal_dir not in sys.path:
    sys.path.insert(0, camcal_dir)
import BoardInfo
# ────────────────────────────────────────────────────

def get_next_index(dest_root):
    existing = [d for d in os.listdir(dest_root) if d.startswith("c_")]
    if not existing:
        return 0
    return max(int(d.split("_",1)[1]) for d in existing) + 1

# most importatant and non functional at the moment
# capture code
def capture_workflow():
    import cv2
    import numpy as np
    import shutil
    from cv2 import aruco
    from GrayCodesWindow import getImageIteration, destroyW
    from CaptureImage import capture_and_save_image

    print("\n--- Starting Gray Code Capture ---")

    is_calib = input("Capture calibration sets? (y/n): ").strip().lower().startswith("y")
    if is_calib:
        parent = "Calib3"
        total_sets = int(input("How many calibration sets? ").strip())
    else:
        parent = input("Enter object folder name: ").strip()
        total_sets = 1

    dest_root = os.path.join(SCRIPT_DIR, "captures", parent)
    os.makedirs(dest_root, exist_ok=True)

    # opens the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera.")
        return

    # warms up camera so it captures a real image, not blacked out
    print("→ Warming up camera...")
    for _ in range(30):
        cap.read()
    cv2.waitKey(500)

    for idx in range(total_sets):
        print(f"\n=== Preparing to capture set {idx}/{total_sets - 1} ===")

        # only do CharUco check if this is a calibration set, not for object sets
        if is_calib:
            while True:
                print("→ Checking CharUco corners with white image…")

                white_img = 255 * np.ones((1080, 1920), dtype=np.uint8)
                cv2.namedWindow("CheckWindow", cv2.WINDOW_NORMAL)
                cv2.setWindowProperty("CheckWindow", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.moveWindow("CheckWindow", 1920, 0)  # adjust as needed
                cv2.imshow("CheckWindow", white_img)
                cv2.waitKey(1000)

                # throw away few frames before reading
                for _ in range(5):
                    cap.read()
                ret, frame = cap.read()

                if not ret:
                    print("❌ Failed to capture frame")
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = aruco.detectMarkers(gray, BoardInfo.arucoDict)

                if ids is None or len(ids) == 0:
                    print("❌ No markers detected")
                    input("Fix board or focus and press ENTER to retry…")
                    continue

                _, cc, ci = aruco.interpolateCornersCharuco(corners, ids, gray, BoardInfo.charucoBoard)
                found = 0 if ci is None else len(ci)
                total_needed = BoardInfo.charucoBoard.getChessboardCorners().shape[0]

                print(f"→ Found {found}/{total_needed} corners.")

                if found == total_needed:
                    print("✅ 48/48 corners found, proceeding with capture")
                    break
                else:
                    print("⚠️ Not enough corners, adjust scene and try again")
                    input("Fix scene and press ENTER to retry…")
        else:
            print("Skipping CharUco detection (object scan mode)")

        # temp folder from tutorial
        temp_folder = os.path.join(SCRIPT_DIR, "CaptureImages", "captures", "charucotest")
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
        os.makedirs(temp_folder)

        # runs the Gray Code capture, not actual file, simply took Gray code and placed here
        print("→ Capturing full Gray Code pattern set...")
        FirstIteration = True
        for imgnr in getImageIteration(FirstIteration):
            capture_and_save_image(os.path.join(temp_folder, imgnr + ".jpg"))
        destroyW()

        # save w.jpg from the last known good frame (for calibration) 
        if is_calib:
            cv2.imwrite(os.path.join(temp_folder, "w.jpg"), frame)
        else:
            # takes a white image again just to save
            white_img = 255 * np.ones((1080, 1920), dtype=np.uint8)
            cv2.imshow("CheckWindow", white_img)
            cv2.waitKey(1000)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(os.path.join(temp_folder, "w.jpg"), frame)

        # moves to the final folder 
        dst = os.path.join(dest_root, f"c_{idx}")
        os.makedirs(dst, exist_ok=True)
        for fn in os.listdir(temp_folder):
            shutil.move(os.path.join(temp_folder, fn), os.path.join(dst, fn))

        print(f"Set saved -> {dst}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ All captures complete.")



# decode code flow, should be completed
def decode_workflow():
    print("\n--- Decode Workflow ---")

    # 1) For calibration sets, will decode each set found in Calib3
    if input("Decode calibration sets in 'Calib3'? (y/n): ").strip().lower().startswith("y"):
        base_cal = os.path.join(SCRIPT_DIR, "captures", "Calib3")
        if not os.path.isdir(base_cal):
            print(f"ERROR: '{base_cal}' not found, skipping calibration decode.")

        else:
            for entry in sorted(os.listdir(base_cal)):
                if not entry.startswith("c_"):
                    continue
                fld = os.path.join(base_cal, entry)
                print(f"[Calibration] Decoding {entry}…")
                subprocess.run([
                    "python",
                    os.path.join(SCRIPT_DIR, "DecodeGrayImages", "DecodeGray.py"),
                    fld
                ], check=True)
        print(" Calibration decode complete for Calib3 folder \n")

    # 2) for object set
    if input("Decode object sets? (y/n): ").strip().lower().startswith("y"):
        obj_name = input("Enter object folder name: ").strip()
        base_obj = os.path.join(SCRIPT_DIR, "captures", obj_name)   # master/captures/objectname

        # if name not found redo
        if not os.path.isdir(base_obj):
            print(f"ERROR: '{base_obj}' not found, skipping object decode.")

        else:
            for entry in sorted(os.listdir(base_obj)):
                if not entry.startswith("c_"):
                    continue
                fld = os.path.join(base_obj, entry)
                print(f"[Object '{obj_name}'] Decoding {entry}…")
                subprocess.run([
                    "python",
                    os.path.join(SCRIPT_DIR, "DecodeGrayImages", "DecodeGray.py"),
                    fld
                ], check=True)
            print(" Object decoding complete \n")

    print("All decoding finished, or name typed incorrectly")


# make sure in the main that the number of directries is correect master > cameracalibration > main.py
# calibration code flow, should be completed
def calibration_workflow():
    print("\n--- Begining calibration code ---")
    subprocess.run([
        "python", os.path.join(SCRIPT_DIR, "CameraCalibration", "main.py")
    ], check=True)
    print(" Calibration complete ")


# reproject code flow, should be completed
def reprojection_workflow():
    print("\n--- Begining reproject code ---")
    subprocess.run([
        "python", os.path.join(SCRIPT_DIR, "ReprojectImage", "main.py")
    ], check=True)
    print(" Reprojection completed ")


# npz reader function, simply will display the instrinsic M and return value
def NPZReader_workflow():
    print("\n--- Reading calibration matrix from .npz file ---")
    try:
        subprocess.run([
            sys.executable,
            os.path.join(SCRIPT_DIR, "CameraCalibration", "Calibration photos", "npzReader.py")
        ], check=True)
        print("✅ Matrix printed successfully.")
    except Exception as e:
        print(f"❌ Failed to run npzReader.py: {e}")

# main
def main():
    done = False

    # interative UI IO that can run starting from any at any point of the Gray Code process
    while (not done):
        print("\n=== STRUCTURED-LIGHT ORCHESTRATOR FILE ===")
        print("1) Capture new images, for object or calibration")
        print("2) Decode, for calibration sets or object set")
        print("3) Calibrate, using current sets inside structure--light-master")
        print("4) Reproject, using object file and current .npz from camera calibration")
        print("5) Read the npz file, visualize intrinsic matrix and return value")
        print("6) Exit")

        c = input("Choose 1–6: ").strip()

        if c=="1":
            capture_workflow()
        elif c=="2":
            decode_workflow()
        elif c=="3":
            calibration_workflow()
        elif c=="4":
            reprojection_workflow()
        elif c =="5":
            NPZReader_workflow()
        elif c=="6":
            exit()
        else:
            print("invalid.")

    print("\nAll done! 🎉")

if __name__=="__main__":
    main()
