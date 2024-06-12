from cv2 import aruco
 
#A4_shape = 2480, 3508
#A4_shape_1cm_margin = 2280, 3308
#outshape = 2380*4, 3308*4
#8.5_shape = 2550, 3300
#8.5_shape_1cm_margin = 2350, 3100
#outshape = 2380*4, 3308*4

dpmm = 40
A4_shape = 216, 279
A4_shape_margin = A4_shape[0]-10, A4_shape[1] - 20
outshape = A4_shape[0]*dpmm, A4_shape[1]*dpmm
desired_block_size_mm = 30
desired_aurco_size_mm = 21
desired_gap_size_mm = 5
 
blocksx = A4_shape[0]//desired_block_size_mm
blocksy = A4_shape[1]//desired_block_size_mm
 
aurcoDict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
charucoBoard = aruco.CharucoBoard((blocksx,
                                         blocksy),
                                         desired_block_size_mm,
                                         desired_aurco_size_mm,
                                         aurcoDict)
 
 
blocksx2 = (A4_shape[0]+desired_gap_size_mm)//(desired_block_size_mm+desired_gap_size_mm)
blocksy2 = (A4_shape[1]+desired_gap_size_mm)//(desired_block_size_mm+desired_gap_size_mm)
 
arucoBoard = aruco.GridBoard((blocksx2,
                                    blocksy2),
                                    desired_block_size_mm,
                                    desired_gap_size_mm,
                                    aurcoDict)