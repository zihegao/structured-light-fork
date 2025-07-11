# UnwrapPhase.py
# 
# this file unwraps the results from the decode phase step that exists in the same folder as this script
# the reulst should be a fine gradient image that will then be use by calibration and reproject phase steps

import os
import cv2
import numpy as np
from collections import deque

# spatial unwrapping function using BFS
def unwrap_phase(wrapped_phase, invalid_mask):
    H, W = wrapped_phase.shape
    unwrapped = np.full_like(wrapped_phase, np.nan, dtype=np.float32)
    visited = np.zeros_like(wrapped_phase, dtype=bool)

    # find seed pixel (first valid pixel)
    ys, xs = np.where(invalid_mask == 0)
    if len(xs) == 0:
        print("[WARN] No valid pixels found!")
        return unwrapped

    seed_y, seed_x = ys[0], xs[0]
    unwrapped[seed_y, seed_x] = wrapped_phase[seed_y, seed_x]
    visited[seed_y, seed_x] = True

    queue = deque()
    queue.append((seed_y, seed_x))

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        y, x = queue.popleft()
        phi_ref = unwrapped[y, x]

        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                if visited[ny, nx] or invalid_mask[ny, nx] != 0:
                    continue

                phi_wrapped = wrapped_phase[ny, nx]
                N = np.round((phi_ref - phi_wrapped) / (2 * np.pi))
                phi_unwrapped = phi_wrapped + 2 * np.pi * N

                unwrapped[ny, nx] = phi_unwrapped
                visited[ny, nx] = True
                queue.append((ny, nx))

    return unwrapped

def process_all_folders(captures_dir):
    for name in sorted(os.listdir(captures_dir)):
        cdir = os.path.join(captures_dir, name)
        if not (os.path.isdir(cdir) and name.startswith("c_")):
            continue

        print(f"[INFO] Unwrapping phase in {name}")

        wrapped_x = cv2.imread(os.path.join(cdir, "wrapped_phase_x.tiff"), cv2.IMREAD_UNCHANGED)
        wrapped_y = cv2.imread(os.path.join(cdir, "wrapped_phase_y.tiff"), cv2.IMREAD_UNCHANGED)
        invH = cv2.imread(os.path.join(cdir, "out_InvalidImageH.tiff"), cv2.IMREAD_GRAYSCALE)
        invV = cv2.imread(os.path.join(cdir, "out_InvalidImageV.tiff"), cv2.IMREAD_GRAYSCALE)

        if wrapped_x is None or wrapped_y is None or invH is None or invV is None:
            print(f"[WARN] Missing inputs for {name} → skipping")
            continue

        # Unwrap
        unwrapped_x = unwrap_phase(wrapped_x, invH)
        unwrapped_y = unwrap_phase(wrapped_y, invV)

        # Save
        cv2.imwrite(os.path.join(cdir, "x_unwrapped.tiff"), unwrapped_x.astype(np.float32))
        cv2.imwrite(os.path.join(cdir, "y_unwrapped.tiff"), unwrapped_y.astype(np.float32))

        print(f"[OK] Saved unwrapped phase for {name}")

if __name__ == "__main__":
    root = os.path.abspath(os.path.dirname(__file__))
    target = os.path.join("..", "captures", "Calib3")
    if not os.path.isdir(target):
        print(f"[ERROR] Invalid folder: {target}")
    else:
        process_all_folders(target)
