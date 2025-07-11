# DecodeGray.py
# ─────────────────────────────────────────────────────────────────────────────
# Structured Light Gray Code Decoder
# This script decodes a set of Gray code structured light images captured with
# a camera-projector system. It outputs binary and gray code images, direct and
# indirect illumination maps, and invalid pixel masks for both horizontal and
# vertical directions.
# 
# Requires: Captured image sets including h*, v*, ih*, iv*, b, w
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import numpy as np
import os

# Constants for configuration
DEFAULT_IMAGE_FORMAT = ".jpg"   # File extension for input images
NUMBER_OF_BITS = 10              # Number of Gray code bits used per direction
mVal = 15                        # Minimum direct light intensity to trust pixel

# ─────────────────────────────────────────────────────────────────────────────
# Function: get_gray_code
# Converts a Gray code value to its binary equivalent
# Input: gray - integer Gray code value
# Output: corresponding binary code value
# ─────────────────────────────────────────────────────────────────────────────
def get_gray_code(gray):
    gray ^= (gray >> 8)
    gray ^= (gray >> 4)
    gray ^= (gray >> 2)
    gray ^= (gray >> 1)
    return gray

# ─────────────────────────────────────────────────────────────────────────────
# Function: get_is_lit
# Determines whether a pixel is lit based on threshold comparisons
# Inputs:
#   - Ld: direct light component
#   - Lg: global/ambient light component
#   - pVal: pixel value from pattern
#   - ipVal: pixel value from inverse pattern
#   - epsalon: margin for lighting variation
# Output: (is_lit: bool, invalid_flag: 0 or 255)
# ─────────────────────────────────────────────────────────────────────────────
def get_is_lit(Ld, Lg, pVal, ipVal, epsalon=5):
    if (Ld > Lg + epsalon) and (pVal > ipVal + epsalon):
        return True, 0
    if (Ld > Lg + epsalon) and (pVal + epsalon < ipVal):
        return False, 0
    if (pVal + epsalon < Ld) and (ipVal > Lg + epsalon):
        return False, 0
    if (pVal > Lg + epsalon) and (ipVal + epsalon < Ld):
        return True, 0
    return False, 255

# ─────────────────────────────────────────────────────────────────────────────
# Function: load_images
# Loads the h/v and inverse h/v images from a folder
# Inputs:
#   - current_dir: path to directory with input images
#   - img_format: file extension (e.g., ".jpg")
# Output:
#   - tuple of lists: (success_flag, h, v, ih, iv, ph, pv, pih, piv)
# ─────────────────────────────────────────────────────────────────────────────
def load_images(current_dir, img_format):
    h_imgs, v_imgs = [], []
    ih_imgs, iv_imgs = [], []
    ph, pv, pih, piv = [], [], [], []

    for i in range(NUMBER_OF_BITS):
        h_img = cv2.imread(os.path.join(current_dir, f"h{i}{img_format}"), cv2.IMREAD_GRAYSCALE)
        v_img = cv2.imread(os.path.join(current_dir, f"v{i}{img_format}"), cv2.IMREAD_GRAYSCALE)
        ih_img = cv2.imread(os.path.join(current_dir, f"ih{i}{img_format}"), cv2.IMREAD_GRAYSCALE)
        iv_img = cv2.imread(os.path.join(current_dir, f"iv{i}{img_format}"), cv2.IMREAD_GRAYSCALE)

        if h_img is None or v_img is None:
            print("OpenCV could not open some files. Please ensure the following files exist:")
            print(f"{os.path.join(current_dir, f'h{i}{img_format}')}")
            print(f"{os.path.join(current_dir, f'v{i}{img_format}')}")
            print(f"{os.path.join(current_dir, f'ih{i}{img_format}')}")
            print(f"{os.path.join(current_dir, f'iv{i}{img_format}')}")
            return False, None, None, None, None

        h_imgs.append(h_img)
        v_imgs.append(v_img)
        ih_imgs.append(ih_img)
        iv_imgs.append(iv_img)
        ph.append(h_img)
        pv.append(v_img)
        pih.append(ih_img)
        piv.append(iv_img)

    return True, h_imgs, v_imgs, ih_imgs, iv_imgs, ph, pv, pih, piv

# ─────────────────────────────────────────────────────────────────────────────
# Function: main
# Main decoding pipeline for Gray Code images
# Steps:
#   - Load images and reference black/white
#   - Compute per-pixel direct/global light and threshold masks
#   - Decode Gray code to binary indices in x and y
#   - Save resulting maps (gray, binary, invalid, illumination)
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import sys
    current_dir = "."
    img_format = DEFAULT_IMAGE_FORMAT

    if len(sys.argv) >= 2:
        current_dir = sys.argv[1]
        print(f"The following directory will be used to search for the captured Images: {current_dir}")
    if len(sys.argv) > 2:
        img_format = sys.argv[2]
        print(f"The following image format will be used: {img_format}")

    success, h_imgs, v_imgs, ih_imgs, iv_imgs, ph, pv, pih, piv = load_images(current_dir, img_format)
    if not success:
        return

    # Load black and white reference images
    b_img = cv2.imread(os.path.join(current_dir, f"b{img_format}"), cv2.IMREAD_GRAYSCALE)
    w_img = cv2.imread(os.path.join(current_dir, f"w{img_format}"), cv2.IMREAD_GRAYSCALE)
    if b_img is None or w_img is None:
        print("OpenCV could not open some files. Please ensure the following files exist:")
        print(f"{os.path.join(current_dir, f'b{img_format}')}")
        print(f"{os.path.join(current_dir, f'w{img_format}')}")
        return

    # Initialize output arrays for both directions
    bin_image_h = np.zeros(b_img.shape, dtype=np.uint16)
    gray_image_h = np.zeros(b_img.shape, dtype=np.uint16)
    invalid_image_h = np.zeros(b_img.shape, dtype=np.uint8)
    bin_image_v = np.zeros(b_img.shape, dtype=np.uint16)
    gray_image_v = np.zeros(b_img.shape, dtype=np.uint16)
    invalid_image_v = np.zeros(b_img.shape, dtype=np.uint8)
    direct_image = np.zeros(b_img.shape, dtype=np.uint8)
    indirect_image = np.zeros(b_img.shape, dtype=np.uint8)

    pb, pw = b_img, w_img

    # ─────────────────────────────────────────────────────────────────────────
    # Decode each pixel
    # ─────────────────────────────────────────────────────────────────────────
    for px in range(b_img.size):
        # Estimate high/low light from the 4 most significant bits
        i_high = max(ph[9].flat[px], pih[9].flat[px], pv[9].flat[px], piv[9].flat[px],
                     ph[8].flat[px], pih[8].flat[px], pv[8].flat[px], piv[8].flat[px],
                     ph[7].flat[px], pih[7].flat[px], pv[7].flat[px], piv[7].flat[px],
                     ph[6].flat[px], pih[6].flat[px], pv[6].flat[px], piv[6].flat[px])
        i_low = min(ph[9].flat[px], pih[9].flat[px], pv[9].flat[px], piv[9].flat[px],
                    ph[8].flat[px], pih[8].flat[px], pv[8].flat[px], piv[8].flat[px],
                    ph[7].flat[px], pih[7].flat[px], pv[7].flat[px], piv[7].flat[px],
                    ph[6].flat[px], pih[6].flat[px], pv[6].flat[px], piv[6].flat[px])

        # Estimate Ld (direct) and Lg (global) light
        b_inv = float(pw.flat[px]) / (pw.flat[px] + pb.flat[px])
        Ld = (i_high - i_low) * b_inv
        Lg = 2.0 * (i_high - Ld) * b_inv

        direct_image.flat[px] = np.clip(Ld, 0, 255)
        indirect_image.flat[px] = np.clip(Lg, 0, 255)

        # Initialize pixel values
        gray_image_h.flat[px] = 0
        gray_image_v.flat[px] = 0
        bin_image_h.flat[px] = 0
        bin_image_v.flat[px] = 0

        if Ld < mVal:
            invalid_image_h.flat[px] = 255
            invalid_image_v.flat[px] = 255
            continue

        # Decode horizontal bits
        val_to_add = 1
        invalid_image_h.flat[px] = 0
        for i in range(NUMBER_OF_BITS - 1, -1, -1):
            is_lit, invalid_flag = get_is_lit(Ld, Lg, ph[i].flat[px], pih[i].flat[px])
            if is_lit:
                gray_image_h.flat[px] += val_to_add
            if invalid_flag:
                invalid_image_h.flat[px] = invalid_flag
            val_to_add <<= 1
        bin_image_h.flat[px] = get_gray_code(gray_image_h.flat[px])

        # Decode vertical bits
        val_to_add = 1
        invalid_image_v.flat[px] = 0
        for i in range(NUMBER_OF_BITS - 1, -1, -1):
            is_lit, invalid_flag = get_is_lit(Ld, Lg, pv[i].flat[px], piv[i].flat[px])
            if is_lit:
                gray_image_v.flat[px] += val_to_add
            if invalid_flag:
                invalid_image_v.flat[px] = invalid_flag
            val_to_add <<= 1
        bin_image_v.flat[px] = get_gray_code(gray_image_v.flat[px])

    # Save all output images
    cv2.imwrite(os.path.join(current_dir, "out_DirectImage.tiff"), direct_image)
    cv2.imwrite(os.path.join(current_dir, "out_IndirectImage.tiff"), indirect_image)
    cv2.imwrite(os.path.join(current_dir, "out_BinImageH.tiff"), bin_image_h)
    cv2.imwrite(os.path.join(current_dir, "out_GrayImageH.tiff"), gray_image_h)
    cv2.imwrite(os.path.join(current_dir, "out_InvalidImageH.tiff"), invalid_image_h)
    cv2.imwrite(os.path.join(current_dir, "out_BinImageV.tiff"), bin_image_v)
    cv2.imwrite(os.path.join(current_dir, "out_GrayImageV.tiff"), gray_image_v)
    cv2.imwrite(os.path.join(current_dir, "out_InvalidImageV.tiff"), invalid_image_v)

# Script entry point
if __name__ == "__main__":
    main()
