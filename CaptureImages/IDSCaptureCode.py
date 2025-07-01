# IDSCaptureCode.py
#
# Structured light capture using IDS camera and phase-shifting patterns.
# Enhanced timing: dummy flushes before all captures (white, black, vertical, horizontal),
# extra settle for first vertical phase image, allowing the projector more time to render v0.
# Horizontal extra delay can be tuned later if needed.
# Naming convention: outputs as w.png, b.png, v0..vN.png, h0..hN.png
#"

import os
import cv2
import numpy as np
import time
import structuredlight as sl
from interface import IDSinterface

# --------------------------------------------------------------------
# Configuration and varaibles 
WINDOW_NAME         = "StructuredLight"
PROJ_WIDTH          = 1920
PROJ_HEIGHT         = 1080
NUM_PHASE_IMAGES    = 3
REF_SETTLE_MS       = 1500  # ms for white/black
PHASE_SETTLE_MS     = 1000  # ms base for each phase pattern
VERT_EXTRA_DELAY    = 4000  # ms extra before first vertical fringe (tuned up)
HORIZ_EXTRA_DELAY   = 0     # ms extra before first horizontal fringe
EXPOSURE_US         = 25000
GAIN                = 1.0
BASE_OUTPUT_DIR     = "captures/Calib2/"

# --------------------------------------------------------------------

def wait_for_render(ms):
    """
    Process GUI events and wait without processing key presses.
    """
    end = time.time() + ms/1000.0
    while time.time() < end:
        cv2.waitKey(1)


# add at top of your config section:
FLUSH_BEFORE_COUNT = 5   # how many stale frames to drop before settle
FLUSH_AFTER_COUNT  = 2   # how many transitional frames to drop after settle

def ids_show_and_capture(cam, img_pattern, settle_ms):
    """
    Show img_pattern full-screen, flush buffers before/after, then capture one fresh frame.
    """
    # 1) display the pattern
    cv2.imshow(WINDOW_NAME, img_pattern)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.waitKey(1)

    # 2) drop any old frames before the pattern even comes up
    for _ in range(FLUSH_BEFORE_COUNT):
        cam.capture()

    # 3) wait for the projector/display to actually update
    wait_for_render(settle_ms)
    cv2.waitKey(1)

    # 4) drop any lingering “black” frames that snuck in during the transition
    for _ in range(FLUSH_AFTER_COUNT):
        cam.capture()

    # 5) grab the one good frame
    frame = cam.capture()

    # convert to single‐channel if needed
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 1:
        return frame[:, :, 0]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



def main():
    # ensure output directory
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    # initialize camera
    cam = IDSinterface()
    cam.select_and_start_device()
    cam.set_gain(GAIN)
    cam.set_exposure_time(EXPOSURE_US)
    try:
        cam.set_aoi(0, 0, PROJ_WIDTH, PROJ_HEIGHT)
    except AttributeError:
        print("Warning: set_aoi() not supported; using sensor default.")

    # setup projection window
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
    except:
        pass

    # display instruction and wait for ENTER
    instr = np.zeros((PROJ_HEIGHT, PROJ_WIDTH), np.uint8)
    cv2.putText(instr, "Position window and press ENTER to start",
                (50, PROJ_HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    print("Position the projection window and press ENTER in it to begin capture.")
    while True:
        cv2.imshow(WINDOW_NAME, instr)
        if cv2.waitKey(100) == 13:
            break

    # warm-up dummy frame
    print("Warm-up: rendering dummy frame (2s) to prep display...")
    dummy = np.zeros((PROJ_HEIGHT, PROJ_WIDTH), np.uint8)
    cv2.imshow(WINDOW_NAME, dummy)
    cv2.waitKey(2000)
    _ = cam.capture(); _ = cam.capture()

    # --- White reference ---
    print("Capturing white reference (w.png)")
    white = 255 * np.ones((PROJ_HEIGHT, PROJ_WIDTH), np.uint8)
    w_img = ids_show_and_capture(cam, white, REF_SETTLE_MS)
    cv2.imwrite(os.path.join(BASE_OUTPUT_DIR, "w.png"), w_img)

    # --- Black reference ---
    print("Capturing black reference (b.png)")
    black = np.zeros((PROJ_HEIGHT, PROJ_WIDTH), np.uint8)
    b_img = ids_show_and_capture(cam, black, REF_SETTLE_MS)
    cv2.imwrite(os.path.join(BASE_OUTPUT_DIR, "b.png"), b_img)

    # prepare phase patterns
    ps = sl.PhaseShifting(num=NUM_PHASE_IMAGES)

    # --- Vertical phase patterns ---
    print(f"Capturing {NUM_PHASE_IMAGES} vertical phase patterns (v0..v{NUM_PHASE_IMAGES-1}.png)")
    for idx, pat in enumerate(sl.transpose(ps.generate((PROJ_HEIGHT, PROJ_WIDTH)) )):
        # extra delay for the first vertical pattern
        settle = PHASE_SETTLE_MS + (VERT_EXTRA_DELAY if idx == 0 else 0)
        img = ids_show_and_capture(cam, pat, settle)
        fname = f"v{idx}.png"
        cv2.imwrite(os.path.join(BASE_OUTPUT_DIR, fname), img)
        print(f"Saved {fname}")

    # --- Horizontal phase patterns ---
    print(f"Capturing {NUM_PHASE_IMAGES} horizontal phase patterns (h0..h{NUM_PHASE_IMAGES-1}.png)")
    for idx, pat in enumerate(ps.generate((PROJ_WIDTH, PROJ_HEIGHT))):
        # no extra horizontal delay for now
        settle = PHASE_SETTLE_MS + (HORIZ_EXTRA_DELAY if idx == 0 else 0)
        img = ids_show_and_capture(cam, pat, settle)
        fname = f"h{idx}.png"
        cv2.imwrite(os.path.join(BASE_OUTPUT_DIR, fname), img)
        print(f"Saved {fname}")

    # cleanup
    cam.stop_acquisition()
    cv2.destroyAllWindows()
    print("Capture complete. All files in", BASE_OUTPUT_DIR)

if __name__ == "__main__":
    main()
