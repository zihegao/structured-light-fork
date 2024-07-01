import numpy as np

"""f = np.load("calculated_Kinect_Stereo_matrix.npz")
for key in ['retval', 'cameraMatrix1', 'distCoeffs1', 'cameraMatrix2', 'distCoeffs2']:
    print(f[key])"""
 
"""f = np.load("Kinect_ir_Mtx.npz")
for key in ['retval', 'cameraMatrix', 'distCoeffs']:
    print(f[key])"""
 
"""f = np.load("Kinect_rgb_Mtx.npz")
for key in ['retval', 'cameraMatrix', 'distCoeffs']:
    print(f[key])"""
 
f = np.load("calculated_cams_matrix.npz")
for key in ['retval', 'cameraMatrix1', 'distCoeffs1', 'cameraMatrix2', 'distCoeffs2']:
    print(f[key])