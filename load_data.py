import numpy as np
import laspy
import open3d as o3d
import cv2
import warnings
from pathlib import Path

def load_pngs_from_folder(mask_folder):
    image_paths = sorted([
        p for p in Path(mask_folder).iterdir()
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
    POP = np.loadtxt(pop_path)
    SOP = np.loadtxt(sop_path)
    intrinsics = np.loadtxt(camera_intrinsics, delimiter=",")
    
    return POP, SOP, intrinsics


def load_point_cloud(pc_folder):
    '''
    Load exactly one LAS/LAZ point cloud as an Open3D tensor point cloud.

    Current assumptions:
    - return_number is required for downstream occlusion handling
    - RGB is optional
    - intensity is assumed present in the current datasets

    Accepted for now:
    - XYZ, Intensity, ReturnNumber
    - XYZ, Intensity, RGB, ReturnNumber
    '''
    
    pc_paths = sorted([
        p for p in Path(pc_folder).iterdir()
        if p.suffix.lower() == ".laz" or p.suffix.lower() == ".las"
    ])

    if len(pc_paths) == 1:

        p = pc_paths[0]
        las = laspy.read(p)
        dims = {d.lower() for d in las.point_format.dimension_names}
        has_rgb = {"red", "green", "blue"} <= dims

        if "return_number" not in dims: # fatal error, not return number --> exit
            raise SystemExit("ERROR!!! Point Cloud does not contain Return Number. Exiting.")
        
        if has_rgb:
            points_tensor = np.vstack((las.x, las.y, las.z)).transpose().astype(np.float32)
            color_tensor = np.vstack((las.red, las.green, las.blue)).transpose().astype(np.float32) / 65535.0
            intensity_tensor = np.vstack((las.intensity)).reshape(-1, 1).astype(np.float32)
            return_number = np.vstack((las.return_number)).reshape(-1,1).astype(np.int8)
            pc_name = p.stem
    
            device = o3d.core.Device("CPU:0")
            pcd = o3d.t.geometry.PointCloud(device)
            pcd.point.positions = o3d.core.Tensor(points_tensor, device=device)
            pcd.point.colors = o3d.core.Tensor(color_tensor,  device=device)
            pcd.point.intensity = o3d.core.Tensor(intensity_tensor, device=device)
            pcd.point.return_number = o3d.core.Tensor(return_number, device=device)
        else: # warning: will only load xyz, I, N 
            warnings.warn("Warning: Point Cloud does not contain RGB. Loading as X, Y, Z, I, N")
            
            points_tensor = np.vstack((las.x, las.y, las.z)).transpose().astype(np.float32)
            intensity_tensor = np.vstack((las.intensity)).reshape(-1, 1).astype(np.float32)
            return_number = np.vstack((las.return_number)).reshape(-1,1).astype(np.int8)
            pc_name = p.stem
    
            device = o3d.core.Device("CPU:0")
            pcd = o3d.t.geometry.PointCloud(device)
            pcd.point.positions = o3d.core.Tensor(points_tensor, device=device)
            pcd.point.intensity = o3d.core.Tensor(intensity_tensor, device=device)
            pcd.point.return_number = o3d.core.Tensor(return_number, device=device)

        return pcd, pc_name
    
    elif len(pc_paths) == 0: # fatal error, no point cloud --> Exit
        raise SystemExit("ERROR!! Please add a point cloud. Exiting.")

    else: # fatal error, too many point clouds --> Exit
        raise SystemExit("ERROR!! Please include ONLY 1 point cloud. Exiting")