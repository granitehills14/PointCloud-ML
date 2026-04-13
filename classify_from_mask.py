if __name__ == "__main__":
    import os
    import numpy as np
    import laspy
    from numpy.linalg import inv
    import open3d as o3d
    from types import SimpleNamespace
    import cv2
    from pathlib import Path

    from load_data import load_pngs_from_folder, load_matrices, load_point_cloud
    from write_data import write_pcd_to_laz
    from classify_2d_to_3d import glcs_to_socs, classify_point_cloud

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
    POP, SOP, intrinsics = load_matrices(paths['pop'], paths['sop'], paths['intrinsics'])

    pc_glcs, pc_name = load_point_cloud(paths['data'])

    masks, mask_paths = load_pngs_from_folder(paths['masks'])

    #%% Step 3: Transfer classification from masks to point cloud
    pc_socs = glcs_to_socs(pc_glcs, POP, SOP)

    pc_glcs_classified = classify_point_cloud(pc_socs, pc_glcs, masks, mask_paths, intrinsics, num_classes, paths['matrices'] )

    #%% Step  4: Write classified point cloud to disk
    write_pcd_to_laz(pc_glcs_classified, paths['out'], SCANPOS, pc_name)