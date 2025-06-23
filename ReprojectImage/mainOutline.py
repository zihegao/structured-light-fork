# Phase-Based Structured Light 3D Reconstruction (Reproject Main)
# ─────────────────────────────────────────────────────────────────────────────
# This script reconstructs 3D point clouds using phase decoded projector-camera
# correspondences and stereo calibration results. It loads matched coordinate maps,
# triangulates depth from geometry, assigns color, and exports point clouds in .ply/.bin.
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import numpy as np
from open3d import open3d
import os
import struct

# Parameters
imgfmt = ".jpg"
baseReadPath = "./captures/"
datasets = ["WaterBottle"]  # List of object folders to process
projector_resolution = (1920, 1080)

# Load stereo calibration results (previously computed)
CalibAtribs = np.load("./camera_calibration_out/calculated_cams_matrix.npz")
retval = CalibAtribs["retval"]
cameraMatrix1 = CalibAtribs["cameraMatrix1"]
distCoeffs1 = CalibAtribs["distCoeffs1"]
cameraMatrix2 = CalibAtribs["cameraMatrix2"]
distCoeffs2 = CalibAtribs["distCoeffs2"]
R = CalibAtribs["R"]
T = CalibAtribs["T"][:, 0]  # Flatten translation vector
E = CalibAtribs["E"]
F = CalibAtribs["F"]
newcameramtx_camera = CalibAtribs["newcameramtx_camera"]
roi_camera = CalibAtribs["roi_camera"]
newcameramtx_proj = CalibAtribs["newcameramtx_proj"]
roi_proj = CalibAtribs["roi_proj"]
invCamMtx = CalibAtribs["invCamMtx"]
invProjMtx = CalibAtribs["invProjMtx"]

# Process each object folder
for dataset in datasets:
    for captureFolder in os.listdir(baseReadPath + dataset):
        if not os.path.isdir(baseReadPath + dataset + "/" + captureFolder):
            continue
        path = baseReadPath + dataset + "/" + captureFolder + "/"

        # Load required images
        img = cv2.imread(path + "w.jpg")
        validV = cv2.imread(path + "out_InvalidImageV.tiff", cv2.IMREAD_GRAYSCALE)
        validH = cv2.imread(path + "out_InvalidImageH.tiff", cv2.IMREAD_GRAYSCALE)
        coordsV = cv2.imread(path + "out_BinImageH.tiff", cv2.IMREAD_ANYDEPTH + cv2.IMREAD_GRAYSCALE)  # projector x
        coordsH = cv2.imread(path + "out_BinImageV.tiff", cv2.IMREAD_ANYDEPTH + cv2.IMREAD_GRAYSCALE)  # projector y

        # Find valid pixels (not masked by invalid maps)
        indImg1 = np.indices(coordsH.shape, coordsH.dtype)
        indImg1 = indImg1[:, np.logical_and(validV == 0, validH == 0)]  # valid camera coords

        colors = img[indImg1[0], indImg1[1]]  # RGB at valid pixels

        # Corresponding projector pixels from decoded maps
        indImg2 = np.vstack((coordsH[indImg1[0], indImg1[1]], coordsV[indImg1[0], indImg1[1]]))
        indImg1[[0, 1]] = indImg1[[1, 0]]  # switch to (x,y)
        indImg2[[0, 1]] = indImg2[[1, 0]]

        indImg1 = indImg1.T.astype(np.float64)
        indImg2 = indImg2.T.astype(np.float64)

        # Convert image points to rays in homogeneous space
        LPts = cv2.convertPointsToHomogeneous(cv2.undistortPoints(indImg1[None,], cameraMatrix1, distCoeffs1, R=R))[:, 0].T
        RPts = cv2.convertPointsToHomogeneous(cv2.undistortPoints(indImg2[None,], cameraMatrix2, distCoeffs2))[:, 0].T

        # Triangulate depth using stereo geometry (angle-based method)
        TLen = np.linalg.norm(T)
        NormedL = LPts / np.linalg.norm(LPts, axis=0)
        alpha = np.arccos(np.dot(-T, NormedL) / TLen)
        beta = np.arccos(np.dot(T, RPts) / (TLen * np.linalg.norm(RPts, axis=0)))
        gamma = np.pi - alpha - beta
        P_len = TLen * np.sin(beta) / np.sin(gamma)
        Pts = NormedL * P_len  # Reconstructed 3D points

        # Swap R and B color channels for Open3D
        colors[:, [0, 2]] = colors[:, [2, 0]]

        # Create full-resolution point cloud
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(Pts.T)
        pcd.colors = open3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        open3d.io.write_point_cloud(path + "capturedPointCloud_" + captureFolder + ".ply", pcd)

        # Save to binary format
        with open(path + "capturedPointCloud_" + captureFolder + ".bin", "wb") as fp:
            fp.write(struct.pack("i", len(Pts.T)))
            for pt in Pts.T:
                fp.write(struct.pack("i", 3) + struct.pack("d", pt[0]) + struct.pack("d", pt[1]) + struct.pack("d", pt[2]))

        # Downsample the point cloud to reduce size
        downpcd = pcd.voxel_down_sample(voxel_size=0.5)
        open3d.io.write_point_cloud(path + "downsampled_capturedPointCloud_" + captureFolder + ".ply", downpcd)

        # Save downsampled point cloud to .bin
        with open(path + "downsampled_capturedPointCloud_" + captureFolder + ".bin", "wb") as fp:
            fp.write(struct.pack("i", len(downpcd.points)))
            for pt in downpcd.points:
                fp.write(struct.pack("i", 3) + struct.pack("d", pt[0]) + struct.pack("d", pt[1]) + struct.pack("d", pt[2]))

        # Optionally filter spatial bounds to clean up stray points
        pts_hold = np.array(downpcd.points)
        colors_hold = np.array(downpcd.colors)
        filterLocs = np.logical_and(
            np.logical_and(pts_hold[:, 2] < 1567537, pts_hold[:, 2] > -8836782),  # z range
            np.logical_and(
                np.logical_and(pts_hold[:, 0] < 3992105, pts_hold[:, 0] > -791900),  # x range
                np.logical_and(pts_hold[:, 1] < 2826662, pts_hold[:, 1] > -221705)   # y range
            )
        )

        # Apply filtering
        pts_hold = pts_hold[filterLocs]
        colors_hold = colors_hold[filterLocs]

        # Save filtered point cloud to .ply and .bin
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(pts_hold)
        pcd.colors = open3d.utility.Vector3dVector(colors_hold)
        open3d.io.write_point_cloud(path + "filtered__capturedPointCloud__" + captureFolder + ".ply", pcd)

        with open(path + "filtered__capturedPointCloud__" + captureFolder + ".bin", "wb") as fp:
            fp.write(struct.pack("i", len(pcd.points)))
            for pt in pcd.points:
                fp.write(struct.pack("i", 3) + struct.pack("d", pt[0]) + struct.pack("d", pt[1]) + struct.pack("d", pt[2]))
