from cv2 import aruco
 
#A4_shape = 2480, 3508
#A4_shape_1cm_margin = 2280, 3308
#outshape = 2380*4, 3308*4
#8.5_shape = 2550, 3300
#8.5_shape_1cm_margin = 2350, 3100
#outshape = 2380*4, 3308*4

# dots per milimeter, being the print resolution
dpmm = 40

# paper size in mm using (width, height)
A4_shape = 216, 279

# created a print safe layout by removing the marigin
A4_shape_margin = A4_shape[0]-10, A4_shape[1] - 20

# resolution of the output image, uses dots for height and width
outshape = A4_shape[0]*dpmm, A4_shape[1]*dpmm

# marker layout specs in milimeters
desired_block_size_mm = 30
desired_aurco_size_mm = 21
desired_gap_size_mm = 5

# ====== staet of ChArUco board setup ======

# number of blocks to fit in the page, for width and heigh in x/y directions
blocksx = A4_shape[0]//desired_block_size_mm
blocksy = A4_shape[1]//desired_block_size_mm

# use a predefined 4x4 ArUco marker dictionary
aurcoDict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

# Create a ChArUco board (checkerboard with embedded ArUco markers)
charucoBoard = aruco.CharucoBoard((blocksx,
                                         blocksy),
                                         desired_block_size_mm,
                                         desired_aurco_size_mm,
                                         aurcoDict)
 

# ===== ArUco grid board setup =====

# Number of markers that fit considering spacing between them
blocksx2 = (A4_shape[0]+desired_gap_size_mm)//(desired_block_size_mm+desired_gap_size_mm)
blocksy2 = (A4_shape[1]+desired_gap_size_mm)//(desired_block_size_mm+desired_gap_size_mm)

# Create a regular ArUco grid board (no checkerboard)
arucoBoard = aruco.GridBoard((blocksx2,
                                    blocksy2),
                                    desired_block_size_mm,
                                    desired_gap_size_mm,
                                    aurcoDict)
