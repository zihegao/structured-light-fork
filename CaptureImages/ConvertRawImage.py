# code will convert .cr2 files to .tiff files, uses rawpy and opencv
# can run a input arg script for further processing
# to usage: python script.py <image_directory> <input_format> [optional_script]

import os
import sys
import cv2
import rawpy
from subprocess import Popen

DETACHED_PROCESS = 0x00000008

# command line args
saveDir = sys.argv[1]
imgFmt = sys.argv[2]

# if cr2 file this converts it
if imgFmt == ".cr2":
    print("Converting images in "+saveDir)

    for file in os.listdir(saveDir):
        print(file)
        f, ext = os.path.splitext(file)
        if ext == ".cr2":
            print("Converting: "+file)
            img = cv2.cvtColor(rawpy.imread(saveDir+file).postprocess(), cv2.COLOR_RGB2BGR)

            # then its saved as a .tiff file (which is the formated neeeded for later)
            cv2.imwrite(saveDir+f+".tiff", img)
else:
    print("Skipping Conversion")

# if there is a 4th arg run as a detatched subprocess
if len(sys.argv)>2 and sys.argv[3] is not None:
    print([sys.argv[3], saveDir, imgFmt])
    Popen([sys.argv[3] +" "+saveDir[:-1] +" "+ imgFmt if imgFmt != ".cr2" else ".tiff"], shell=True,
                    stdin=None, stdout=None,
                    stderr=None, close_fds=True)
cv2.waitKey(10000)
