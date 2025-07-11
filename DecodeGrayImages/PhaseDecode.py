""" 
# ReadMe script for 
PhaseDecode.py

this is a decode 3 step (120*) phase shift sequence 

to work the file needs the following from the CaptureCode.py with PhaseShift = True
    b.jpg (black)
    w.jpg (white)
    h0.jpg, h1.jpg, h2.jpg    # the 3 horizontal phase-shift images at 0* 120* 240*
    v0.jpg, v1.jpg, v2.jpg    # the 3 vertical   phase-shift images at 0* 120* 240*

outputs will be written in the same folder, just added to the current set
    x_index.tiff              # uint16 map of unwrapped horizontal projector index ∈ [0..PROJ_W-1]
    y_index.tiff              # uint16 map of unwrapped vertical   projector index ∈ [0..PROJ_H-1]
    out_InvalidImageH.tiff    # uint8 mask (0=valid,255=invalid) for horizontal decode
    out_InvalidImageV.tiff    # uint8 mask (0=valid,255=invalid) for vertical decode

To run (keep in mind no command line args, just run inside VScode)
  1) Place PhaseDecode.py and all 8 source .jpg files in the same folder.
  2) run this file
  3) The four TIFF outputs will appear next to the .jpg files to then be used in the next step, PhaseCalibration
"""

import os
import sys
import cv2
import numpy as np

# needed for BoardInfo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CameraCalibration')))
import BoardInfo

# config to setup
PROJ_W = 1920
PROJ_H = 1080
mVal = 40               # direct light threshold
BrightThresh = 70       # brightness threshold

def decode_phase(images):
    # phase shifts, -120*, 0*, +120*
    delta = np.array([-2*np.pi/3, 0, 2*np.pi/3], dtype=np.float32)
    A = np.array([
        [1, np.cos(delta[0]), np.sin(delta[0])],
        [1, np.cos(delta[1]), np.sin(delta[1])],
        [1, np.cos(delta[2]), np.sin(delta[2])]
    ], dtype=np.float32)
    Ainv = np.linalg.inv(A)

    H, W = images[0].shape
    stack = np.stack(images, axis=-1).reshape(-1, 3)
    coeffs = stack @ Ainv.T
    a0, a1, a2 = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2]

    # direct and average brightness
    Ld = 2 * np.sqrt(a1 ** 2 + a2 ** 2)
    B0 = 3 * a0

    # wrapped phase formula
    num = np.sqrt(3) * (images[0].reshape(-1) - images[2].reshape(-1))
    den = (2 * images[1].reshape(-1) - images[0].reshape(-1) - images[2].reshape(-1))
    phi = np.arctan2(num, den)
    phi[phi < 0] += 2 * np.pi

    return phi.reshape(H, W), Ld.reshape(H, W), B0.reshape(H, W)

def process_folder(base_capture_dir):
    for name in sorted(os.listdir(base_capture_dir)):
        cdir = os.path.join(base_capture_dir, name)
        if not (os.path.isdir(cdir) and name.startswith("c_")):
            continue

        b_img = cv2.imread(os.path.join(cdir, "b.jpg"), cv2.IMREAD_GRAYSCALE)
        w_img = cv2.imread(os.path.join(cdir, "w.jpg"), cv2.IMREAD_GRAYSCALE)
        if b_img is None or w_img is None:
            print(f"[WARN] {name}: Missing b.jpg or w.jpg → skipping")
            continue

        h_imgs = [cv2.imread(os.path.join(cdir, f"h{i}.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float32) for i in range(3)]
        v_imgs = [cv2.imread(os.path.join(cdir, f"v{i}.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float32) for i in range(3)]
        if any(im is None for im in h_imgs + v_imgs):
            print(f"[WARN] {name}: Missing h or v images → skipping")
            continue

        phi_h, Ld_h, B0_h = decode_phase(h_imgs)
        phi_v, Ld_v, B0_v = decode_phase(v_imgs)

        invH = np.zeros_like(b_img, dtype=np.uint8)
        invV = np.zeros_like(b_img, dtype=np.uint8)
        invH[(Ld_h < mVal) | (B0_h < BrightThresh)] = 255
        invV[(Ld_v < mVal) | (B0_v < BrightThresh)] = 255

        # saves the  wrapped phase and masks
        cv2.imwrite(os.path.join(cdir, "x_index.tiff"), np.round((phi_h / (2*np.pi)) * (PROJ_W - 1)).astype(np.uint16))
        cv2.imwrite(os.path.join(cdir, "y_index.tiff"), np.round((phi_v / (2*np.pi)) * (PROJ_H - 1)).astype(np.uint16))
        cv2.imwrite(os.path.join(cdir, "wrapped_phase_x.tiff"), phi_h.astype(np.float32))
        cv2.imwrite(os.path.join(cdir, "wrapped_phase_y.tiff"), phi_v.astype(np.float32))
        cv2.imwrite(os.path.join(cdir, "out_InvalidImageH.tiff"), invH)
        cv2.imwrite(os.path.join(cdir, "out_InvalidImageV.tiff"), invV)

        print(f"[OK] Decoded wrapped phase for {name}")

if __name__ == "__main__":
    root = os.path.abspath(os.path.dirname(__file__))
    target = os.path.join("..", "captures", "Calib3")

    if not os.path.isdir(target):
        print(f"[ERROR] Invalid folder: {target}")
        sys.exit(1)

    print(f"[START] Phase decoding in {target} …")
    process_folder(target)
    print("[DONE] Wrapped phase decoding complete.")
