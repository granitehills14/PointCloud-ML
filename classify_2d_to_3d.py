#%% Step 0: Import Packages
import numpy as np
from numpy.linalg import inv
import open3d as o3d
import cv2

def glcs_to_socs(pc_glcs, POP, SOP):
       
    print(f"Transforming from GLCS to SOCS")

    # Define each intermediate point cloud
    pc_socs = o3d.t.geometry.PointCloud(pc_glcs.clone())

    # transform each point into the camera's reference frame
    pc_socs.transform(inv(POP)).transform(inv(SOP)) # glcs > prcs > SOCS
    
    return pc_socs


def classify_point_cloud(pc_socs, pc_glcs, masks, mask_paths, intrinsics, num_classes, matrices):
    N = pc_socs.point.positions.shape[0] # number of points
    device = pc_socs.point.positions.device # is the point cloud on CPU or GPU

    pc_cmcs = o3d.t.geometry.PointCloud(pc_socs.clone()) # clone pc to be transformed later
    
    # break apart the intrinsics file to build what we need for later
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]  

    MM = np.loadtxt(f"{matrices}/mounting.dat" , delimiter='\t') # the mounting matrix for the camera
    inv_MM = inv(MM)

    px_vals = np.full((N, len(mask_paths)), 255, dtype=np.int32)

    for j, img in enumerate(masks):
        # for each image, transform pc into the camera's frame of reference, then project the points onto the image
        print(f"Applying masks from {mask_paths[j].stem}")
        z_rot = np.loadtxt(f"{matrices}/{mask_paths[j].stem.removesuffix('_mask')}.dat") # the z-rotation matrix for img
        
        if img.shape[0] > img.shape[1]: # check that the image is landscape
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) # rotate the image
        
        h, w = img.shape[:2] # assign h and w from img dimensions

        pc_temp = o3d.t.geometry.PointCloud(pc_socs.clone())
        pc_temp.transform(inv(z_rot)).transform(inv_MM) # Transform the point cloud into the camera's reference frame

        pts = pc_temp.point.positions.numpy() # make the point cloud into something through which I can loop.

        x = pts[:,0]
        y = pts[:,1]
        z = pts[:,2]

        return_numbers = pc_temp.point.return_number.numpy().reshape(-1)

        finite_xyz = np.isfinite(pts).all(axis=1)
        camera_front = (z > 0)
        first_single_returns = (return_numbers == 1)
        valid_3d = finite_xyz & camera_front & first_single_returns

        idx = np.where(valid_3d)[0]

        valid_x = x[idx]
        valid_y = y[idx]
        valid_z = z[idx]

        u = ((fx * valid_x) / valid_z) + cx
        v = ((fy * valid_y) / valid_z) + cy

        finite_uv = np.isfinite(u) & np.isfinite(v)

        idx = idx[finite_uv]
        
        n = np.floor(u[finite_uv] + 0.5).astype(np.int32)
        m = np.floor(v[finite_uv] + 0.5).astype(np.int32)

        frustum = (0 <= n) & (n < w) & (0 <= m) & (m < h)

        valid_point_indices = idx[frustum]

        sampled_values = img[m[frustum], n[frustum]]

        px_vals[valid_point_indices, j] = sampled_values
    
    pc_cmcs.point.px_vals = o3d.core.Tensor(px_vals, dtype=o3d.core.int32, device=device)

    class_votes = np.zeros((N, num_classes + 1), dtype=np.int32) # make a (N, num_classes) Numpy array to hold the class_votes

    for c in range(num_classes + 1):
        class_votes[:,c] = np.count_nonzero(px_vals == c, axis=1) # count how many of each class exist in px_vals
    
    labels = np.full(N, 255, dtype=np.int32) # create the labels Numpy array to take the values from the voting routine, default value is 255 ("no data")
    has_votes = np.count_nonzero(class_votes, axis=1) > 0 # tests each point for whether or not it has any class values
    labels[has_votes] = np.argmax(class_votes[has_votes], axis=1).astype(np.int32) # for the points that have class votes (i.e. project to the image sensor), write the column index of the class with the most votes to the labels array. If the point has only 0s for it's class votes, it gets skipped and the label value remains 255. In the case where there is a tie the smaller index gets chosen by this. I had an idea of using kNN as a tie-breaker but that might be too complex for now.

    pc_cmcs.point.class_votes = o3d.core.Tensor(
        # (N, num_classes) tensor storing the class votes for each point
        class_votes, dtype=o3d.core.int32, device=device
    )

    pc_cmcs.point.classification = o3d.core.Tensor(
        # (N, ) tensor storing the derived classification value for each point
        labels, dtype = o3d.core.int32, device = device
        )

    pc_glcs_classified = o3d.t.geometry.PointCloud(pc_glcs.clone())
    pc_glcs_classified.point.classification = pc_cmcs.point.classification.clone()

    return pc_glcs_classified