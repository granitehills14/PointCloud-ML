'''
Data loading script
Development version for implementing SAM3

Branched from Main: 21 April 2026
'''
import json
import numpy as np
import laspy
import open3d as o3d
import cv2
from pathlib import Path

def set_up_scanpos_folders(project_path):
    scanpos_folders = sorted([
        f for f in Path(project_path).iterdir()
        if f.is_dir() and f.name.startswith("ScanPos")
        ])
    return scanpos_folders

def load_config(config_path, config_json):
    '''
    Load the config JSON
        settings  
            Project Name
            ScanPos
            Number of Classes
        sam3
            Key words for the text prompts
    '''
    print("Loading config file...")
    config_file = Path(config_path) / config_json

    with open(config_file, 'r') as f:
        config = json.load(f)

    if config is None:
        raise SystemExit("FATAL ERROR! NO CONFIG FILE FOUND. EXITING.")

    settings = config["settings"]
    sam3 = config["sam3"]
    neighborhood = config["neighborhood"]
    
    if settings["USING_SAM"]:
        num_classes = len(sam3["prompts"])
    else: 
        num_classes = settings["num_classes"]
    
    return settings, sam3, neighborhood, num_classes 


def load_raw_imagery(jpeg_folder):
    '''
    Load raw images from folder
    Store all raw images as a numpy array in a list. Track valid image names and paths via valid_paths.
    '''
    print("Loading raw imagery...")
    image_paths = sorted([
        p for p in Path(jpeg_folder).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg"}
        ])

    images = []
    valid_paths = []

    if len(image_paths) == 0:
        raise SystemExit("FATAL ERROR! NO IMAGES FOUND! EXITING")
    
    for p in image_paths:
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            print(f"Warning: could not read {p}")
            continue

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        images.append(img)
        valid_paths.append(p)
    
    return images, valid_paths


def load_pngs_from_folder(mask_folder):
    print("Loading image masks...")
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
    print("Loading POP and SOP matrices...")
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
        if p.suffix.lower() in {".laz", ".las"}
    ])

    if len(pc_paths) == 1:

        p = pc_paths[0]
        las = laspy.read(p)
        dims = {d.lower() for d in las.point_format.dimension_names}
        has_rgb = {"red", "green", "blue"} <= dims

        if "return_number" not in dims: # fatal error, not return number --> exit
            raise SystemExit("ERROR!!! Point Cloud does not contain Return Number. Exiting.")
        
        if has_rgb:
            print(f"Loading {p} with RGB...")
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
            print("Warning: Point Cloud does not contain RGB. Loading as X, Y, Z, I, N")
            
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