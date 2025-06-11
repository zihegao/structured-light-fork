# TestCorners.py
# this file is meant to be run to test if all corners can be seen currently
# run code in terminal as "enter" msut be pressed
# does not expect any command line inputs, rather camera only 
# make sure scene is not black, and the charucoboard has some illumination

import cv2
import BoardInfo
import numpy as np
from cv2 import aruco

def main():
    # opens conencted camera
    cam_index = 0  # if dif indexx change
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"Error: Cannot open camera index {cam_index}")
        return

    # warms up the camera
    print("Warming up camera...")
    for _ in range(30):
        cap.read()

    # when ready get UI from user
    input("Press ENTER to capture an image from the camera...")
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Failed to capture image.")
        return

    image = frame
    # show the captured image briefly
    cv2.imshow("Captured Image", image)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    # detects aruco markers
    corners_list, ids, _ = aruco.detectMarkers(image, BoardInfo.arucoDict)
    if ids is None or len(ids) == 0:
        print("No ArUco markers detected.")
        return

    # interpolate ChArUco corners
    _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
        markerCorners=corners_list,
        markerIds=ids,
        image=image,
        board=BoardInfo.charucoBoard
    )

    if charuco_ids is None or len(charuco_ids) == 0:
        print("No ChArUco corners detected.")
        return

    detected = len(charuco_ids)
    expected = BoardInfo.charucoBoard.getChessboardCorners().shape[0]

    print(f"Detected ChArUco corners: {detected} / {expected}")
    if detected >= expected:
        print("All corners detected! ✅")
    else:
        print(f"Missing {expected - detected} corners. ⚠️")

if __name__ == '__main__':
    main()