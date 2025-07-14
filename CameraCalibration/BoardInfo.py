from cv2 import aruco
 
#paper_shape = 2480, 3508
#A4_shape_1cm_margin = 2280, 3308
#outshape = 2380*4, 3308*4
#8.5_shape = 2550, 3300
#8.5_shape_1cm_margin = 2350, 3100
#outshape = 2380*4, 3308*4

dpmm = 40 # Dots per millimeter used for scaling
paper_shape = 216, 279 # US letter paper size in mm
paper_margin = 10 # margin in mm
# A4_shape_margin = paper_shape[0]-10, paper_shape[1] - 20  
outshape = (paper_shape[0]-paper_margin)*dpmm, (paper_shape[1]-paper_margin)*dpmm
desired_block_size_mm = 30
desired_aruco_size_mm = 21
# desired_gap_size_mm = 5 #gap between markers of Aruco board
 
blocksx = paper_shape[0]//desired_block_size_mm # Horizontal number of blocks
blocksy = paper_shape[1]//desired_block_size_mm # Vertical number of blocks
 
arucoDict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250) #4x4=16bit maerker dictionary with 250 markers

charucoBoard = aruco.CharucoBoard(
                                         (blocksx, blocksy), # Number of markers in X,Y coordinate system
                                         desired_block_size_mm, # Size of a square in mm
                                         desired_aruco_size_mm, #size of Aruco marker
                                         arucoDict) 
 
 
# blocksx2 = (paper_shape[0]+desired_gap_size_mm)//(desired_block_size_mm+desired_gap_size_mm)
# blocksy2 = (paper_shape[1]+desired_gap_size_mm)//(desired_block_size_mm+desired_gap_size_mm)
 
# arucoBoard = aruco.GridBoard(
#                               (blocksx2, blocksy2), # Number of markers in (X, Y)
#                               desired_block_size_mm, #Marker size
#                               desired_gap_size_mm,  #marker gap size
#                               arucoDict)
