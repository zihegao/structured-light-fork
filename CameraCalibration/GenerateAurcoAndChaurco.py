import cv2
import numpy as np
from cv2 import aruco
import BoardInfo

print(f'{cv2.__version__}')

# calculate the physical width and height of the ChArUco board in millimeters
aw=BoardInfo.blocksx*BoardInfo.desired_block_size_mm
ah=BoardInfo.blocksy*BoardInfo.desired_block_size_mm

# generate ChArUco board image scaled to dpmm (dots per millimeter) resolution
img = aruco.CharucoBoard.generateImage(BoardInfo.charucoBoard, (aw * BoardInfo.dpmm, ah * BoardInfo.dpmm), marginSize=BoardInfo.paper_margin*BoardInfo.dpmm)

# save the ChArUco board image to charucoBoard.png
cv2.imwrite("charucoBoard.png", img)
print(f"Generated charucoBoard.png, /n please print it out with paper size {BoardInfo.papersize}mm x {BoardInfo.papersize}mm", marginSize = )

# # calculate the physical width and height of the ArUco grid board including gaps
# aw2=BoardInfo.blocksx2*BoardInfo.desired_block_size_mm+(BoardInfo.blocksx2-1)*BoardInfo.desired_gap_size_mm
# ah2=BoardInfo.blocksy2*BoardInfo.desired_block_size_mm+(BoardInfo.blocksy2-1)*BoardInfo.desired_gap_size_mm

# # generate ArUco grid board image with the calculated size
# img = BoardInfo.arucoBoard.generateImage((aw2*BoardInfo.dpmm, ah2*BoardInfo.dpmm))

# # save the ArUco board image to the file arucoBoard.png
# cv2.imwrite("arucoBoard.png", img)


