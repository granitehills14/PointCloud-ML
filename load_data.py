import os
import numpy as np
import laspy
import open3d as o3d
import cv2
from pathlib import Path

def load_pngs_from_folder(mask_folder):
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


def load_matrices(pop_path, sop_path, camera_intrinsics ):
    POP = np.loadtxt(pop_path, delimiter=',')
    SOP = np.loadtxt(sop_path, delimiter=',')
    intrinsics = np.loadtxt(camera_intrinsics, delimiter=',')
    
    return POP, SOP, intrinsics


def load_point_cloud(pc_folder): # there should only ever be 1 point cloud in the directory. Not sure if this is coded properly for that.
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