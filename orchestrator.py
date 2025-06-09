# =============================================================================
# File:  orchestrator.py
#
# this file is used as a so called "main main"
# intended for the user to run from terminal (where IO can happen)
# script asks user information such as starting point, object file name, number of calibration sets
# if decoding takes a while/shows /0 error, if code still running ignore and continue to let code run
# individual files can still be run if following tutoiral doc, 
# 
# HOW TO RUN
# no inputs from command line, inputs from CaptureCode.py where images are taken
# outputs, .tiff files will be in calibration folders and .ply files will be alongside c_# inside *object* file master>captures>object
# =============================================================================


import os
import shutil
import subprocess

# ────────────────────────────────────────────────────────────────
# helper stuff

def get_next_index(dest_root):
    existing = [d for d in os.listdir(dest_root) if d.startswith("c_")]
    if not existing:
        return 0
    nums = [int(d.split("_",1)[1]) for d in existing]
    return max(nums) + 1

# ────────────────────────────────────────────────────
# capture section (1)

def capture_workflow():
    is_obj = input("Are these object images? (y/n): ").strip().lower().startswith("y")
    parent = "Calib3" if not is_obj else input("Enter object folder name: ").strip()
    total_sets = int(input(f"How many sets will you capture? ").strip())
    
    dest_root = os.path.join("Captures", parent)
    os.makedirs(dest_root, exist_ok=True)

    for set_num in range(total_sets):
        print(f"\n[Step: Capture set {set_num+1}/{total_sets}]  Press ENTER when ready to run CaptureCode.py")
        input()

        # call the capture code
        subprocess.run(["python", "CaptureImages/CaptureCode.py"], check=True)

        # asks user what name is typed into the CaptureCode.py
        run_folder = input("  ↳ What folder name did you enter into CaptureCode.py? ").strip()
        src = os.path.join("CaptureImages", "captures", run_folder)
        idx = get_next_index(dest_root)
        dst = os.path.join(dest_root, f"c_{idx}")
        os.makedirs(dst)

        # move and clean 
        for fn in os.listdir(src):
            shutil.move(os.path.join(src, fn), dst)
        os.rmdir(src)

        print(f"[Step: Move] Captures moved → {dst}")
        ok = input("  ↳ Confirm all corners detected OK? (y/n): ").strip().lower()
        if not ok.startswith("y"):
            print("    → Retaking this same set index…")
            set_num -= 1
        else:
            print("    → Good, moving on.")

# ─────────────────────────────────────────────────────
# decode section (2)

def decode_workflow():
    print("\n--- Decode Calibration Sets ---")
    total_cal = int(input("How many calibration sets in 'Calib3'? ").strip())
    base_cal = os.path.join("Captures", "Calib3")

    for i in range(total_cal):
        folder = os.path.join(base_cal, f"c_{i}")
        if not os.path.isdir(folder):
            print(f"[Decode Calib] '{folder}' missing, skipping.")
            continue
        print(f"[Step: Decode Calib c_{i}]  running Gray-code decoder…")
        subprocess.run([
            "python", "DecodeGrayImages/DecodeGray.py", folder
        ], check=True)
        print(f"    → Done c_{i}")

    print("\n--- Decode Object Sets (optional) ---")
    if input("Decode object images too? (y/n): ").strip().lower().startswith("y"):
        parent = input("Enter object folder name: ").strip()
        total_obj = int(input(f"How many sets under '{parent}'? ").strip())
        base_obj = os.path.join("Captures", parent)
        for i in range(total_obj):
            folder = os.path.join(base_obj, f"c_{i}")
            if not os.path.isdir(folder):
                print(f"[Decode Obj] '{folder}' missing, skipping.")
                continue
            print(f"[Step: Decode Obj c_{i}]  running Gray-code decoder…")
            subprocess.run([
                "python", "DecodeGrayImages/cpptopythontest.py", folder
            ], check=True)
            print(f"    → Done c_{i}")

# ──────────────────────────────────────────────────
# reproject section (3)

def reprojection_workflow():
    if input("\nRun point-cloud reprojection now? (y/n): ").strip().lower().startswith("y"):
        print("[Step: Reprojection]  running ReprojectImage/main.py…")
        subprocess.run([
            "python", "ReprojectImage/main.py"
        ], check=True)
        print("    → Reprojection complete.")

# ───────────────────────────────────────────────────────────────────
# main, IO flow begins below

def main():
    print("\n=== STRUCTURED-LIGHT ORCHESTRATOR ===")
    print("1) Capture new images")
    print("2) Decode existing captures")
    print("3) Reproject point clouds")
    choice = input("Select 1, 2, or 3: ").strip()

    if choice == "1":
        capture_workflow()
        if input("\nProceed to decode captures now? (y/n): ").strip().lower().startswith("y"):
            decode_workflow()
        if input("Proceed to reprojection now? (y/n): ").strip().lower().startswith("y"):
            reprojection_workflow()

    elif choice == "2":
        decode_workflow()
        if input("Proceed to reprojection now? (y/n): ").strip().lower().startswith("y"):
            reprojection_workflow()

    elif choice == "3":
        reprojection_workflow()

    else:
        print("Invalid selection—exiting.")

    print("\nAll done! 🎉")

if __name__ == "__main__":
    main()
