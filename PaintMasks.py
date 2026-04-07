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
        'data' : f"pcml/data/poux/DATA/{SCENE}",
        'intrinsics' : f"pcml/data/riegl/CAM/intrinsics.csv",
        'extrinsics' : f"pcml/data/riegl/CAM/extrinsics.csv",
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

    images, image_paths = load_jpegs_from_folder(paths['data'])

    prediction = SimpleNamespace(
        depth=None,
        conf=None,
        intrinsics=np.loadtxt(paths['intrinsics'], delimeter=','),
        extrinsics=None,
        processed_images=images,
    )
    
    # points_3d = data['points_3d']
    # colors_3d = data['colors_3d']

    # print(f"Loaded {len(points_3d)} points, {len(prediction.depth)} frames")
    print(f"Image resolution: {prediction.processed_images[0].shape[:2]}")

    #%% Step 2: Apply HD Masking on Single Image
'''
    sample_image_hd = prediction.processed_images[0]
    hd_mask = paint_mask_hd(sample_image_hd, num_classes=5, target_height=1080)
    print(f"HD mask created: {hd_mask.shape}, Classes: {np.unique(hd_mask)}")

    dsm.visualize_hd_mask(sample_image_hd, hd_mask, "Step 5b: HD Mask on Single Image")

    #%% Step 3: Project to 3D Point Cloud

    # Project single-frame mask to 3D and visualize
    projected_pts, projected_labels = dsm.project_mask_to_3d(
        hd_mask, prediction.depth[0], prediction.intrinsics[0],
        prediction.extrinsics[0], prediction.conf[0], conf_thresh=0.4
    )
    dsm.visualize_projected_mask_3d(projected_pts, projected_labels, "Step 5: Single Frame Mask Projected to 3D")

    #%% Step 6: Complete Workflow — HD masking → multi-frame labels → smart fusion
'''
    print("\n=== Complete Workflow ===")
    num_frames_to_mask = min(5, len(prediction.processed_images))
    print(f"Masking {num_frames_to_mask} frames in HD...")

    hd_multi_masks = mask_multiple_images_hd(
        prediction.processed_images,
        num_images=num_frames_to_mask,
        target_height=None
    )

'''
    frame_indices = list(range(num_frames_to_mask))
    all_labels_3d = dsm.create_full_scene_labels(
        points_3d,
        prediction,
        hd_multi_masks,
        frame_indices,
        conf_thresh=0.4
    )
'''
'''
    camera_positions = dsm.extract_camera_positions(prediction.extrinsics)
    print(f"\nExtracted {len(camera_positions)} camera positions")

    # Apply memory-efficient smart fusion
    fused_labels, camera_distances = dsm.smart_label_fusion(
        points_3d,
        all_labels_3d,
        camera_positions,
        max_distance=0.05,      # 15cm neighborhood
        max_camera_dist=5.0,    # Filter labels beyond 5m
        min_neighbors=3,        # Need 3+ labeled neighbors
        batch_size=100000        # Process 50k points at once (reduce if memory issues)
    )
'''
'''
    # Visualize scene labels before and after fusion
    dsm.visualize_scene_labels(points_3d, all_labels_3d, "Step 8: Scene Labels Before Fusion")
    dsm.visualize_scene_labels(points_3d, fused_labels, "Step 8: Scene Labels After Fusion")

    #%% Step 7: Visualize before/after comparison
    dsm.visualize_fusion_comparison(points_3d, all_labels_3d, fused_labels, camera_positions)

    #%% Step 8: Analyze fusion results
    dsm.analyze_fusion_statistics(all_labels_3d, fused_labels)

    #%% Step 9: Export fused results

    output_fused_ply = os.path.join(paths['results'], "smart_fused_labels-v2.ply")
    dsm.save_point_cloud_as_ply(points_3d, colors_3d, output_fused_ply, labels=fused_labels)

    # Final visualization of exported result
    dsm.visualize_scene_labels(points_3d, fused_labels, "Export: Final Labeled Point Cloud")

    #%% Step 11: Export Labeled Scene as GLB

    glb_output = os.path.join(paths['results'], "semantic_scene.glb")
    dsm.export_labeled_glb(points_3d, fused_labels, prediction, glb_output, camera_size=0.02, estimate_normals=False)
'''