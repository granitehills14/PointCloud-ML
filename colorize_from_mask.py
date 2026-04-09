import os
import numpy as np
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
    'raw' : os.path.join(scanpos_dir, "CAM/images/raw"),
    'masks' : os.path.join(scanpos_dir, "CAM/images/masks"),
    'matrices' : os.path.join(scanpos_dir, "CAM/matrices"),
    'intrinsics' : f"{project_dir}/intrinsics.dat",
    'pop' : f"{project_dir}/POP.dat",
    'sop' : f"{scanpos_dir}/{SCANPOS}.dat"
}

#%% Step 2: Load Data
def load_pngs_from_folder(folder):
    folder = paths['masks']
    image_paths = sorted([
        p for p in folder.iterdir()
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

#%% Step 3: Transformations

def glcs_to_socs(point_cloud):
   
    pc_glcs = o3d.t.io.read_point_cloud(point_cloud) # tensor point cloud in global coordinates
    
    print(f"Transforming {pc_glcs} from GLCS to SOCS")


    # Define each intermediate point cloud
    pc_socs = o3d.t.geometry.PointCloud()
    
    # Define each transformation matrix
    matrices = SimpleNamespace(
        POP = np.loadtext(paths['pop'], delimiter=','),
        SOP = np.loadtext(paths['sop'], delimiter=','),
    )

    # transform each point into the camera's reference frame
    pc_socs = pc_glcs.Transform(inv(matrices.POP)).Transform(inv(matrices.SOP)) # glcs > prcs > SOCS
    
    return pc_socs


def classify_point_cloud(point_cloud_socs, images, point_cloud_glcs):

    pc = o3d.t.io.read_point_cloud(Path(point_cloud_socs)) # read-in the SOCS point cloud

    N = pc.point.positions.shape[0] # number of points
    device = pc.point.positions.device # is the point cloud on CPU or GPU

    pc_cmcs = o3d.t.geometry.PointCloud(pc) # clone pc to be transformed later

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

    camera_intrinsics = np.loadtxt(Path(paths['intrinsics']), delimiter=',') # the camera intrinsics

    # break apart the intrinsics file to build what we need for later
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    dx = 0.00000376
    dy = 0.00000376
    nx = 9504
    ny = 6336
       
    '''K = np.array([fx, 0, cx, 0],
                 [0, fy, cy, 0],
                 [0, 0, 1, 0])'''

    for j, img in enumerate(images):
        # for each image, transform pc into the camera's frame of reference, then project the points onto the image
        z_rot = np.loadtxt(Path(f"{paths['matrices']}/{images[img]}.dat"), delimiter=',') # the z-rotation matrix for img
        MM = np.loadtext(Path(f"{paths['matrices']}/mounting.dat") , delimiter=',') # the mounting matrix for img

        pc_cmcs = o3d.geometry.PointCloud(pc).Transform(inv(z_rot)).Transform(inv(MM)) # Transform the point cloud into the camera's reference frame

        pts = pc_cmcs.points.positions.numpy() # make the point cloud into something through which I can loop.

        for i, (x, y, z) in enumerate(pts): # loop through each point 
            n = ((fx * x) / z) + cx # find the x-pixel the point projects to
            m = ((fy * y) / z) + cy # find the y-pixel the point projects to

            if not np.isnan(img[n,m]): # if there is a pixel value associated with the point
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
    
    pc_glcs = point_cloud_glcs

    pc_glcs.point.classification = o3d.core.Tensor(
        pc_cmcs.point.classification,
        dtype=o3d.core.int32,
        device = device
    )

    return pc_glcs