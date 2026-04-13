#%% Step 0: Import Packages
import os
import numpy as np
import laspy
from numpy.linalg import inv
import open3d as o3d
from scipy.spatial import cKDTree
from types import SimpleNamespace
import trimesh
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import cv2
import matplotlib.pyplot as plt
from types import SimpleNamespace

from interactive_painting import paint_mask_hd, mask_multiple_images_hd
import da_semantic_masking as dsm
import colorMasks as cm

import random
from pathlib import Path

# ===== COLORIZE_FROM_MASK.PY PSEUDO-CODE OUTLINE =====

#%% Step 1: Establish Script-Wide Variables and Paths
num_classes = 5
SCENE = "Basement"
SCANPOS = "ScanPos001"
project_dir = f"pcml/data/riegl/{SCENE}"
scanpos_dir = f"pcml/data/riegl/{SCENE}/{SCANPOS}"

paths = {
    'data' : os.path.join(scanpos_dir, "DATA"),
    'out' : os.path.join(project_dir, "EXPORT"),
    'raw' : os.path.join(scanpos_dir, "CAM/images/raw"),
    'masks' : os.path.join(scanpos_dir, "CAM/images/masks"),
    'matrices' : os.path.join(scanpos_dir, "CAM/matrices"),
    'intrinsics' : f"{project_dir}/intrinsics.dat",
    'pop' : f"{project_dir}/POP.dat",
    'sop' : f"{scanpos_dir}/{SCANPOS}.dat"
}

#%% Step 2: Load Data
def load_pngs_from_folder(mask_folder):
    mask_folder = paths['masks']
    image_paths = sorted([
        p for p in mask_folder.iterdir()
        if p.suffix.lower() == ".png" and not p.stem.endswith("_color")
        ])

    images = []
    valid_paths = []

    for p in image_paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: could not read {p}")
            continue
        images.append(img)
        valid_paths.append(p)
    return images, valid_paths

def load_point_cloud(pc_folder): # there should only ever be 1 point cloud in the directory. Not sure if this is coded properly for that. 
    pc_folder = paths['data']
    pc_paths = sorted([
        p for p in pc_folder.iterdir()
        if p.suffix.lower() == ".laz" or p.suffix.lower() == ".las"
    ])

    pc_name = []

    for p in pc_paths:
        las = laspy.read(Path(f"{pc_folder}/{p}"))
        if las is None:
            print(f"Warning: could not read {p}")
            continue
        points_tensor = np.vstack((las.x, las.y, las.z)).transpose().astype(np.float32)
        color_tensor = np.vstack((las.red, las.green, las.blue)).transpose() / 65535.0
        intensity_tensor = np.vstack((las.intensity)).reshape(-1, 1).astype(np.float32)
        pc_name.append(p.stem)
    
    device = o3d.core.Device("CPU:0")
    pcd = o3d.t.geometry.PointCloud(device)
    pcd.point.positions = o3d.core.Tensor(points_tensor, device=device)
    pcd.point.colors = o3d.core.Tensor(color_tensor,  device=device)
    pcd.point.intensity = o3d.core.Tensor(intensity_tensor, device=device)

    return pcd, pc_name   

point_cloud, point_cloud_name = load_point_cloud(paths['data'])

#%% Step 3: Transformations and Classification
def glcs_to_socs(point_cloud):
   
    pc_glcs = point_cloud # tensor point cloud in global coordinates
    
    print(f"Transforming from GLCS to SOCS")


    # Define each intermediate point cloud
    pc_socs = o3d.t.geometry.PointCloud()
    
    # Define each transformation matrix
    matrices = SimpleNamespace(
        POP = np.loadtxt(paths['pop'], delimiter=','),
        SOP = np.loadtxt(paths['sop'], delimiter=','),
    )

    # transform each point into the camera's reference frame
    pc_socs = pc_glcs.Transform(inv(matrices.POP)).Transform(inv(matrices.SOP)) # glcs > prcs > SOCS
    
    return pc_socs

def classify_point_cloud(point_cloud, images):
    pc_socs = glcs_to_socs(point_cloud) # read-in the SOCS point cloud

    N = pc_socs.point.positions.shape[0] # number of points
    device = pc_socs.point.positions.device # is the point cloud on CPU or GPU

    pc_cmcs = o3d.t.geometry.PointCloud(pc_socs) # clone pc to be transformed later

    pc_cmcs.point.px_vals = o3d.core.Tensor( 
    # (N x len(images)) tensor storing 1 px val per point per image
        np.full((N, len(images)), -1, dtype=np.int32),
        dtype = o3d.core.int32,
        device = device
    )
    
    pc_cmcs.point.class_votes = o3d.core.Tensor( 
    # (N x num_classes) tensor storing the counts of each px_val for each point
        np.full((N, num_classes), 0, dtype=np.int32),
        dtype = o3d.core.int32,
        device = device
    )

    masks, mask_paths = load_pngs_from_folder(paths['masks'])

    camera_intrinsics = np.loadtxt(Path(paths['intrinsics']), delimiter=',') # the camera intrinsics

    # break apart the intrinsics file to build what we need for later
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    dx = 0.00000376
    dy = 0.00000376
    nx = 9504
    ny = 6336

    for j, img in enumerate(masks):
        # for each image, transform pc into the camera's frame of reference, then project the points onto the image
        z_rot = np.loadtxt(Path(f"{paths['matrices']}/{mask_paths[j].stem}.dat"), delimiter=',') # the z-rotation matrix for img
        MM = np.loadtxt(Path(f"{paths['matrices']}/mounting.dat") , delimiter=',') # the mounting matrix for img

        pc_cmcs = o3d.t.geometry.PointCloud(pc_socs).Transform(inv(z_rot)).Transform(inv(MM)) # Transform the point cloud into the camera's reference frame

        pts = pc_cmcs.point.positions.numpy() # make the point cloud into something through which I can loop.

        for i, (x, y, z) in enumerate(pts): # loop through each point 
            n = np.floor((((fx * x) / z) + cx) + 0.5).astype(np.int32) # find the x-pixel the point projects to (assumes all pixel coords are positive)
            m = np.floor((((fy * y) / z) + cy) + 0.5).astype(np.int32) # find the y-pixel the point projects to (assumes all pixel coords are positive)

            if not np.isnan(img[n,m]): # if there is a pixel value associated with the point ***** THIS IS A PROBLEM ******
                pc_cmcs.point.px_vals[i,j] = img[n,m] # push the pixel value to px_vals for each point
                    
    for p, row in enumerate(pc_cmcs.point.px_vals): # for every point
        for c in range(num_classes): # and for each class
            pc_cmcs.point.class_votes[p,c] = np.count_nonzero(row == c) # set the value of class_votes[p,c] to the number of occurances of that class

    for p, row in enumerate(pc_cmcs.point.class_votes): # for each point
        pc_cmcs.point.classification[p] = pc_cmcs.point.class_votes[p,np.int32(np.argmax(row))] # set the classification value to the value of the column index with the most votes. In the case where there is a tie the smaller index gets chosen by this. I had an idea of using kNN as a tie-breaker but that might be too complex for now. 

    votes = np.asarray(pc_cmcs.point.class_votes)
    labels = votes.argmax(axis=1).astype(np.int32)
    
    pc_cmcs.point.classification = o3d.core.Tensor(
        # (N, ) tensor storing the derived classification value for each point
        labels,
        dtype = o3d.core.int32,
        device = device
        )
    
    pc_glcs = point_cloud

    pc_glcs.point.classification = o3d.core.Tensor(
        pc_cmcs.point.classification,
        dtype=o3d.core.int32,
        device = device
    )

    return pc_glcs

#%% Step 4: Write Laz data
def write_pcd_to_laz(pcd):
    precision = 0.00025 # 0.00025m for Riegl TLS, UAS, and MLS instruments

    pcd = classify_point_cloud(point_cloud, paths['masks']) # result of classify_point_cloud()

    points = pcd.point.positions.numpy() # point x, y, z values
    
    if 'colors' in pcd.point:
        colors = pcd.point.colors.numpy() * 65535 # point RGB values
    
    if 'intensity' in pcd.point:
        intensity = pcd.point.intensity.numpy # point intensity values
    
    if 'classification' in pcd.point:
        classification = pcd.point.classification.numpy() # point classification values (from classify_point_cloud())
    
    header = laspy.LasHeader(point_format=6, version="1.4") # set-up header LAS 1.4 PF:6, change if needed (can make these variables if I want)
    header.offsets = np.min(points, axis=0) # no idea what this is doing. I'm pretty sure that's just the smallest x value but I'm not sure
    header.scales = [precision, precision, precision]

    las = laspy.LasData(header)

    las.x = points[:,0]
    las.y = points[:,1]
    las.z = points[:,2]

    las.red = colors[:, 0]
    las.green = colors[:, 1]
    las.blue = colors[:, 2]

    las.intensity = intensity.flatten().astype(np.uint16)

    las.classification = classification.flatten().astype(np.uint8)

    las.write(f"{paths['out']}/{SCANPOS}_{point_cloud_name}.laz")

