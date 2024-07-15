import cv2
import numpy as np
import structuredlight as sl
from matplotlib import pyplot as plt



def loadImages(filename):
    img_list = []
    for i in range(3):
        img = cv2.imread(filename + str(i) + ".jpg")
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_list.append(img_grey)    
    return img_list


width = 1280
height = 720
num_fringes = 4
BaseOutputDir = "../captures/"
capturefolder = "test" #input("Input capture folder:")
OutputDir = BaseOutputDir + capturefolder + "/"

ps = sl.PhaseShifting(num=num_fringes)
im_list = ps.generate((width, height))

imlist_posi_x_cap = loadImages(OutputDir+"h")
imlist_posi_y_cap = loadImages(OutputDir+"v")


img_index_x = ps.decode(imlist_posi_x_cap)
img_index_y = ps.decode(imlist_posi_y_cap)
campoints, prjpoints = sl.getCorrespondencePoints(img_index_x, img_index_y)
print("xy-coord only")
print("campoints: ", campoints.shape)
print(campoints)
print("prjpoints: ", prjpoints.shape)
print(prjpoints)

img_correspondence_x = np.clip(img_index_x/width*255.0, 0, 255).astype(np.uint8)
cv2.imshow("x_corresponnence map", img_correspondence_x)
cv2.imwrite(OutputDir+"x_correspondence.png", img_correspondence_x)

img_correspondence_y = np.clip(img_index_y/width*255.0, 0, 255).astype(np.uint8)
cv2.imshow("y_corresponnence map", img_correspondence_y)
cv2.imwrite(OutputDir+"y_correspondence.png", img_correspondence_y)

img_correspondence = cv2.merge([0.0*np.zeros_like(img_index_x), img_index_x/width, img_index_y/height])
img_correspondence = np.clip(img_correspondence*255.0, 0, 255).astype(np.uint8)
cv2.imshow("x:Green, y:Red", img_correspondence)
cv2.imwrite(OutputDir+"x_ycorrespondence.png", img_correspondence)
cv2.waitKey(0)
cv2.destroyAllWindows()



# img_index = ps.decode(im_list)
# campoints, prjpoints = sl.getCorrespondencePoints(img_index)
# print("xy-coord only")
# print("campoints: ", img_index.shape)
# print(img_index)
# # img_index = img_index[:]
# print("prjpoints: ", prjpoints.shape)
# # print(prjpoints)
# img_correspondence = np.clip(img_index/width*255.0, 0, 255).astype(np.uint8)
# cv2.imshow("corresponnence map", img_correspondence)
# cv2.waitKey(0)


