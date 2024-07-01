import cv2
import numpy as np
import os

# Constants
DEFAULT_IMAGE_FORMAT = ".jpg"
NUMBER_OF_BITS = 10
mVal = 15

# Function to calculate Gray code
def get_gray_code(gray):
    gray ^= (gray >> 8)
    gray ^= (gray >> 4)
    gray ^= (gray >> 2)
    gray ^= (gray >> 1)
    return gray

# Function to determine if a pixel is lit or not
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

# Function to load images
def load_images(current_dir, img_format):
    h_imgs = []
    v_imgs = []
    ih_imgs = []
    iv_imgs = []
    ph = []
    pv = []
    pih = []
    piv = []

    for i in range(NUMBER_OF_BITS):
        # Read horizontal, vertical, inverse horizontal, and inverse vertical images
        h_img = cv2.imread(os.path.join(current_dir, f"h{i}{img_format}"), cv2.IMREAD_GRAYSCALE)
        v_img = cv2.imread(os.path.join(current_dir, f"v{i}{img_format}"), cv2.IMREAD_GRAYSCALE)
        ih_img = cv2.imread(os.path.join(current_dir, f"ih{i}{img_format}"), cv2.IMREAD_GRAYSCALE)
        iv_img = cv2.imread(os.path.join(current_dir, f"iv{i}{img_format}"), cv2.IMREAD_GRAYSCALE)

        # Check if images were loaded successfully
        if h_img is None or v_img is None or ih_img is None or iv_img is None:
            print("OpenCV could not open some files. Please ensure the following files exist:")
            print(f"{os.path.join(current_dir, f'h{i}{img_format}')}")
            print(f"{os.path.join(current_dir, f'v{i}{img_format}')}")
            print(f"{os.path.join(current_dir, f'ih{i}{img_format}')}")
            print(f"{os.path.join(current_dir, f'iv{i}{img_format}')}")
            return False, None, None, None, None

        # Append images to respective lists
        h_imgs.append(h_img)
        v_imgs.append(v_img)
        ih_imgs.append(ih_img)
        iv_imgs.append(iv_img)
        
        ph.append(h_img)
        pv.append(v_img)
        pih.append(ih_img)
        piv.append(iv_img)
    
    return True, h_imgs, v_imgs, ih_imgs, iv_imgs, ph, pv, pih, piv

# Main function
def main():
    import sys
    current_dir = "."
    img_format = DEFAULT_IMAGE_FORMAT

    # Check if command-line arguments are provided for directory and image format
    if len(sys.argv) >= 2:
        current_dir = sys.argv[1]
        print(f"The following directory will be used to search for the captured Images: {current_dir}")
    if len(sys.argv) > 2:
        img_format = sys.argv[2]
        print(f"The following image format will be used: {img_format}")

    # Load images
    success, h_imgs, v_imgs, ih_imgs, iv_imgs, ph, pv, pih, piv = load_images(current_dir, img_format)
    if not success:
        return

    # Read black and white reference images
    b_img = cv2.imread(os.path.join(current_dir, f"b{img_format}"), cv2.IMREAD_GRAYSCALE)
    w_img = cv2.imread(os.path.join(current_dir, f"w{img_format}"), cv2.IMREAD_GRAYSCALE)
    if b_img is None or w_img is None:
        print("OpenCV could not open some files. Please ensure the following files exist:")
        print(f"{os.path.join(current_dir, f'b{img_format}')}")
        print(f"{os.path.join(current_dir, f'w{img_format}')}")
        return

    # Initialize arrays for processed images
    bin_image_v = np.zeros(b_img.shape, dtype=np.uint16)
    gray_image_v = np.zeros(b_img.shape, dtype=np.uint16)
    invalid_image_v = np.zeros(b_img.shape, dtype=np.uint8)

    bin_image_h = np.zeros(b_img.shape, dtype=np.uint16)
    gray_image_h = np.zeros(b_img.shape, dtype=np.uint16)
    invalid_image_h = np.zeros(b_img.shape, dtype=np.uint8)

    direct_image = np.zeros(b_img.shape, dtype=np.uint8)
    indirect_image = np.zeros(b_img.shape, dtype=np.uint8)

    pb = b_img
    pw = w_img

    # Iterate through each pixel
    for px in range(b_img.size):
        # Calculate intensity range
        i_high = max(ph[9].flat[px], pih[9].flat[px], pv[9].flat[px], piv[9].flat[px],
                     ph[8].flat[px], pih[8].flat[px], pv[8].flat[px], piv[8].flat[px],
                     ph[7].flat[px], pih[7].flat[px], pv[7].flat[px], piv[7].flat[px],
                     ph[6].flat[px], pih[6].flat[px], pv[6].flat[px], piv[6].flat[px])
        
        i_low = min(ph[9].flat[px], pih[9].flat[px], pv[9].flat[px], piv[9].flat[px],
                    ph[8].flat[px], pih[8].flat[px], pv[8].flat[px], piv[8].flat[px],
                    ph[7].flat[px], pih[7].flat[px], pv[7].flat[px], piv[7].flat[px],
                    ph[6].flat[px], pih[6].flat[px], pv[6].flat[px], piv[6].flat[px])

        b_inv = float(pw.flat[px]) / (pw.flat[px] + pb.flat[px])

        Ld = (i_high - i_low) * b_inv
        Lg = 2.0 * (i_high - Ld) * b_inv

        # Calculate direct and indirect images
        direct_image.flat[px] = np.clip(Ld, 0, 255)
        indirect_image.flat[px] = np.clip(Lg, 0, 255)

        gray_image_v.flat[px] = 0
        gray_image_h.flat[px] = 0

        bin_image_v.flat[px] = 0
        bin_image_h.flat[px] = 0

        # Check if pixel intensity is below threshold
        if Ld < mVal:
            invalid_image_v.flat[px] = 255
            invalid_image_h.flat[px] = 255
            continue

        # Initialize value to add for each bit
        val_to_add = 1

        # Process horizontal bits
        invalid_image_h.flat[px] = 0
        for i in range(NUMBER_OF_BITS - 1, -1, -1):
            is_lit, invalid_flag = get_is_lit(Ld, Lg, ph[i].flat[px], pih[i].flat[px])
            if is_lit:
                gray_image_h.flat[px] += val_to_add
            if invalid_flag:
                invalid_image_h.flat[px] = invalid_flag
            val_to_add <<= 1
        bin_image_h.flat[px] = get_gray_code(gray_image_h.flat[px])

        # Process vertical bits
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

    # Save processed images
    cv2.imwrite(os.path.join(current_dir, "out_DirectImage.tiff"), direct_image)
    cv2.imwrite(os.path.join(current_dir, "out_IndirectImage.tiff"), indirect_image)

    cv2.imwrite(os.path.join(current_dir, "out_BinImageH.tiff"), bin_image_h)
    cv2.imwrite(os.path.join(current_dir, "out_GrayImageH.tiff"), gray_image_h)
    cv2.imwrite(os.path.join(current_dir, "out_InvalidImageH.tiff"), invalid_image_h)

    cv2.imwrite(os.path.join(current_dir, "out_BinImageV.tiff"), bin_image_v)
    cv2.imwrite(os.path.join(current_dir, "out_GrayImageV.tiff"), gray_image_v)
    cv2.imwrite(os.path.join(current_dir, "out_InvalidImageV.tiff"), invalid_image_v)

if __name__ == "__main__":
    main()

