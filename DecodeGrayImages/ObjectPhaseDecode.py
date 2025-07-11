import os
import cv2
import numpy as np

# ─────────────────────────────────────────────────
# CONFIGURATION
PROJ_W = 1920
PROJ_H = 1080
mVal = 40               # Direct light threshold
BrightThresh = 70       # Brightness threshold
DELTA = np.array([-2*np.pi/3, 0, 2*np.pi/3], dtype=np.float32)
# ─────────────────────────────────────────────────

def decode_phase(images):
    A = np.array([
        [1, np.cos(DELTA[0]), np.sin(DELTA[0])],
        [1, np.cos(DELTA[1]), np.sin(DELTA[1])],
        [1, np.cos(DELTA[2]), np.sin(DELTA[2])]
    ], dtype=np.float32)
    Ainv = np.linalg.inv(A)

    H, W = images[0].shape
    stack = np.stack(images, axis=-1).reshape(-1, 3)
    coeffs = stack @ Ainv.T
    a0, a1, a2 = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2]

    Ld = 2 * np.sqrt(a1 ** 2 + a2 ** 2)
    B0 = 3 * a0

    num = np.sqrt(3) * (images[0].reshape(-1) - images[2].reshape(-1))
    den = (2 * images[1].reshape(-1) - images[0].reshape(-1) - images[2].reshape(-1))
    phi = np.arctan2(num, den)
    phi[phi < 0] += 2 * np.pi

    return phi.reshape(H, W), Ld.reshape(H, W), B0.reshape(H, W)

def unwrap_phase(wrapped_phase, invalid_mask):
    H, W = wrapped_phase.shape
    unwrapped = np.full_like(wrapped_phase, np.nan, dtype=np.float32)
    visited = np.zeros_like(wrapped_phase, dtype=bool)

    ys, xs = np.where(invalid_mask == 0)
    if len(xs) == 0:
        print("[WARN] No valid pixels found!")
        return unwrapped

    from collections import deque
    seed_y, seed_x = ys[0], xs[0]
    unwrapped[seed_y, seed_x] = wrapped_phase[seed_y, seed_x]
    visited[seed_y, seed_x] = True

    queue = deque([(seed_y, seed_x)])
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

def process_object_folder(folder):
    print(f"[INFO] Processing object folder: {folder}")

    for name in sorted(os.listdir(folder)):
        cdir = os.path.join(folder, name)
        if not (os.path.isdir(cdir) and name.startswith("c_")):
            continue

        print(f"[INFO] Working on set: {name}")
        b_img = cv2.imread(os.path.join(cdir, "b.jpg"), cv2.IMREAD_GRAYSCALE)
        w_img = cv2.imread(os.path.join(cdir, "w.jpg"), cv2.IMREAD_GRAYSCALE)
        if b_img is None or w_img is None:
            print(f"[WARN] Missing b.jpg or w.jpg in {name}, skipping")
            continue

        h_imgs = [cv2.imread(os.path.join(cdir, f"h{i}.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float32) for i in range(3)]
        v_imgs = [cv2.imread(os.path.join(cdir, f"v{i}.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float32) for i in range(3)]
        if any(im is None for im in h_imgs + v_imgs):
            print(f"[WARN] Missing phase images in {name}, skipping")
            continue

        phi_h, Ld_h, B0_h = decode_phase(h_imgs)
        phi_v, Ld_v, B0_v = decode_phase(v_imgs)

        invH = np.zeros_like(b_img, dtype=np.uint8)
        invV = np.zeros_like(b_img, dtype=np.uint8)
        invH[(Ld_h < mVal) | (B0_h < BrightThresh)] = 255
        invV[(Ld_v < mVal) | (B0_v < BrightThresh)] = 255

        x_unwrapped = unwrap_phase(phi_h, invH)
        y_unwrapped = unwrap_phase(phi_v, invV)

        cv2.imwrite(os.path.join(cdir, "x_unwrapped.tiff"), x_unwrapped.astype(np.float32))
        cv2.imwrite(os.path.join(cdir, "y_unwrapped.tiff"), y_unwrapped.astype(np.float32))
        cv2.imwrite(os.path.join(cdir, "out_InvalidImageH.tiff"), invH)
        cv2.imwrite(os.path.join(cdir, "out_InvalidImageV.tiff"), invV)

        print(f"[OK] Finished decoding and unwrapping for {name}")

if __name__ == "__main__":
    import sys
    base_path = os.path.abspath(os.path.dirname(__file__))
    target = input("Enter object folder under captures: ").strip()
    obj_folder = os.path.join("..", "captures", target)

    if not os.path.isdir(obj_folder):
        print(f"[ERROR] Invalid folder: {obj_folder}")
    else:
        process_object_folder(obj_folder)
