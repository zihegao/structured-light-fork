
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
import cv2
import numpy as np

# static values, set to own desire
PROJ_W = 1920    # projector width in pixels
PROJ_H = 1080    # projector height in pixels
EPS    = 5       # if (white − black) < EPS, mark pixel invalid # more of how strict value (FIXME may need attention)
###########################################################################

def compute_invalid_mask(b_img, w_img, eps):
    """
    invalid-mask if (w – b) < eps, mark pixel invalid
    Returns a uint8 mask: 0 = valid, 255 = invalid
    """
    diff = w_img.astype(np.float32) - b_img.astype(np.float32)
    inv = np.zeros_like(b_img, dtype=np.uint8)
    inv[diff < eps] = 255
    return inv

def decode_3step(h0, h1, h2):
    """
    given three 8-bit images h0 h1 h2 (because this used 120* phase) compute the wrapped phase per pixel
        φ_wrapped = atan2( sqrt(3)*(h0 – h2), 2*h1 – h0 – h2 )
    Returns φ_wrapped in [0, 2π).
    """
    I0 = h0.astype(np.float32)
    I1 = h1.astype(np.float32)
    I2 = h2.astype(np.float32)

    num = np.sqrt(3.0) * (I0 - I2)
    den = (2.0 * I1 - I0 - I2)

    phi = np.arctan2(num, den)  # in (−π, +π]

    # now map negative angles → [0, 2π)
    phi[phi < 0] += 2 * np.pi
    return phi  # float32 array same shape as h0

def main():
    # use folder with code and images in it already
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_dir = script_dir

    # expected filenames in this same folder
    b_path  = os.path.join(current_dir, "b.jpg")
    w_path  = os.path.join(current_dir, "w.jpg")
    h_paths = [os.path.join(current_dir, f"h{i}.jpg") for i in range(3)]
    v_paths = [os.path.join(current_dir, f"v{i}.jpg") for i in range(3)]

    # 1) loads the black & white reference images
    b_img = cv2.imread(b_path, cv2.IMREAD_GRAYSCALE)
    w_img = cv2.imread(w_path, cv2.IMREAD_GRAYSCALE)
    if b_img is None or w_img is None:
        print(f"ERROR: Could not load 'b.jpg' or 'w.jpg' from {current_dir}")
        return

    # 2) now build invalid masks from (w, b)
    invalid_mask = compute_invalid_mask(b_img, w_img, EPS)
    invalidH = invalid_mask.copy()
    invalidV = invalid_mask.copy()

    # 3) then load the three horizontal phase-shift images h0 h1 h2
    h_imgs = []
    for hp in h_paths:
        img = cv2.imread(hp, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"ERROR: Cannot open '{hp}'")
            return
        h_imgs.append(img)
    h0, h1, h2 = h_imgs

    # 4) move to horiz by decoding horizontal wrapped phase
    phi_x = decode_3step(h0, h1, h2)  # float32 in [0, 2π)

    # 5) convert φ_x → [0..PROJ_W-1] integer index
    x_norm = phi_x / (2 * np.pi)      # map [0,2π) → [0,1)
    x_idx_f = x_norm * (PROJ_W - 1)    # map [0,1) → [0, PROJ_W-1]
    x_index = np.round(x_idx_f).astype(np.uint16)

    # 6) load the three vertical phase-shift images: v0 v1 v2
    v_imgs = []
    for vp in v_paths:
        img = cv2.imread(vp, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"ERROR: Cannot open '{vp}'")
            return
        v_imgs.append(img)
    v0, v1, v2 = v_imgs

    # 7) decode vertical wrapped phase
    phi_y = decode_3step(v0, v1, v2)

    # 8) convert φ_y → [0..PROJ_H-1] integer index
    y_norm = phi_y / (2 * np.pi)
    y_idx_f = y_norm * (PROJ_H - 1)
    y_index = np.round(y_idx_f).astype(np.uint16)

    # 9) save outputs back into the same folder again
    out_x_fn = os.path.join(current_dir, "x_index.tiff")
    out_y_fn = os.path.join(current_dir, "y_index.tiff")
    invH_fn  = os.path.join(current_dir, "out_InvalidImageH.tiff")
    invV_fn  = os.path.join(current_dir, "out_InvalidImageV.tiff")

    cv2.imwrite(out_x_fn, x_index)
    cv2.imwrite(out_y_fn, y_index)
    cv2.imwrite(invH_fn, invalidH)
    cv2.imwrite(invV_fn, invalidV)

    print("Saved:")
    print(f"  {out_x_fn}")
    print(f"  {out_y_fn}")
    print(f"  {invH_fn}")
    print(f"  {invV_fn}")
    print("Decoding complete.")

if __name__ == "__main__":
    main()
