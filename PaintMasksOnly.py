#%% Interactive Tutorial (only runs when executed directly)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    import open3d as o3d
    from scipy.spatial import cKDTree
    from types import SimpleNamespace
    import trimesh
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    import cv2
    import random
    from pathlib import Path

    from interactive_painting_edited import paint_mask_hd, mask_multiple_images_hd
    import da_semantic_masking as dsm

    #%% Step 1: Load Data
    SCENE = "Basement"
    results_dir = f"pcml/data/riegl/RESULTS/{SCENE}"

    paths = {
        'data' : f"pcml/data/riegl/DATA/{SCENE}",
        # 'intrinsics' : f"pcml/data/riegl/CAM/intrinsics.csv",
        # 'extrinsics' : f"pcml/data/riegl/CAM/extrinsics.csv",
        'results' : results_dir,
        'masks' : os.path.join(results_dir, "masks"),
    }
    os.makedirs(paths['masks'], exist_ok=True)

    def load_jpegs_from_folder(folder):
        folder = Path(folder)
        exts = {".jpg", ".jpeg", ".JPG", ".JPEG"}
        image_paths = sorted([p for p in folder.iterdir() if p.suffix in exts])

        images = []
        valid_paths = []

        for p in image_paths:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                print(f"Warning: could not read {p}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            valid_paths.append(p)
        return images, valid_paths

    def colorize_mask(mask):
        colors = np.array([
            [0, 0, 0],       # 0 background
            [255, 0, 0],     # 1
            [0, 255, 0],     # 2
            [0, 0, 255],     # 3
            [255, 255, 0],   # 4
            [255, 0, 255],   # 5
        ], dtype=np.uint8)

        return colors[mask]

    def save_masks(masks, image_paths, output_folder):
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        for mask, img_path in zip(masks, image_paths):
            out_path = output_folder  / f"{img_path.stem}_mask.png"
            cv2.imwrite(str(out_path), mask)
            out_path_color = output_folder  / f"{img_path.stem}_mask_color.png"
            cv2.imwrite(str(out_path_color), colorize_mask(mask))
            print(f"Saved {out_path} and {out_path_color}")


    images, image_paths = load_jpegs_from_folder(paths['data'])

    if not images:
        raise RuntimeError(f"No readable JPEGs found in {paths['data']}")

    prediction = SimpleNamespace(
        depth=None,
        conf=None,
        intrinsics=None, # np.loadtxt(paths['intrinsics'], delimiter=','),
        extrinsics=None,
        processed_images=images,
    )

    print(f"Image resolution: {prediction.processed_images[0].shape[:2]}")

    #%% Step 2: HD masking 
    print("\n=== HD Masking ===")
    num_frames_to_mask = min(5, len(prediction.processed_images))
    print(f"Masking {num_frames_to_mask} frames in HD...")

    hd_multi_masks = mask_multiple_images_hd(
        prediction.processed_images,
        num_images=num_frames_to_mask,
        target_height=1600
    )

    save_masks(hd_multi_masks, image_paths, paths['masks'])