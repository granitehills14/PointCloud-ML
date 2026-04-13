#%% Step 0: Import Packages
import os
import numpy as np
import laspy
from numpy.linalg import inv
import open3d as o3d
import cv2
from pathlib import Path

def glcs_to_socs(pc_glcs, POP, SOP):
       
    print(f"Transforming from GLCS to SOCS")

    # Define each intermediate point cloud
    pc_socs = o3d.t.geometry.PointCloud()
    
    # transform each point into the camera's reference frame
    pc_socs = pc_glcs.Transform(inv(POP)).Transform(inv(SOP)) # glcs > prcs > SOCS
    
    return pc_socs


def classify_point_cloud(pc_socs, pc_glcs, masks, mask_paths, intrinsics, num_classes, matrices):
    N = pc_socs.point.positions.shape[0] # number of points
    device = pc_socs.point.positions.device # is the point cloud on CPU or GPU

    pc_cmcs = o3d.t.geometry.PointCloud(pc_socs) # clone pc to be transformed later

    pc_cmcs.point.px_vals = o3d.core.Tensor( 
    # (N x len(images)) tensor storing 1 px val per point per image
        np.full((N, len(mask_paths)), -1, dtype=np.int32),
        dtype = o3d.core.int32,
        device = device
    )
    
    pc_cmcs.point.class_votes = o3d.core.Tensor( 
    # (N x num_classes) tensor storing the counts of each px_val for each point
        np.full((N, num_classes), 0, dtype=np.int32),
        dtype = o3d.core.int32,
        device = device
    )

    # break apart the intrinsics file to build what we need for later
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    dx = 0.00000376
    dy = 0.00000376
    nx = 9504
    ny = 6336

    for j, img in enumerate(masks):
        # for each image, transform pc into the camera's frame of reference, then project the points onto the image
        z_rot = np.loadtxt(f"{matrices}/{mask_paths[j].stem}.dat", delimiter=',') # the z-rotation matrix for img
        MM = np.loadtxt(f"{matrices}/mounting.dat" , delimiter=',') # the mounting matrix for img

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

    pc_glcs_classified = o3d.t.geometry.PointCloud(pc_glcs)

    pc_glcs_classified.point.classification = o3d.core.Tensor(
        pc_cmcs.point.classification,
        dtype=o3d.core.int32,
        device = device
    )

    return pc_glcs_classified