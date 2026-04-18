#%% Step 0: Import Packages
import numpy as np
from numpy.linalg import inv
import open3d as o3d

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

    pc_cmcs.point.px_vals = o3d.core.Tensor( 
    # (N x len(images)) tensor storing 1 px val per point per image
        np.full((N, len(mask_paths)), -1, dtype=np.int32),
        dtype = o3d.core.int32,
        device = device
    )
    
    # break apart the intrinsics file to build what we need for later
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    MM = np.loadtxt(f"{matrices}/mounting.dat" , delimiter='\t') # the mounting matrix for the camera

    for j, img in enumerate(masks):
        # for each image, transform pc into the camera's frame of reference, then project the points onto the image
        print(f"Applying masks from {mask_paths[j].stem}")
        z_rot = np.loadtxt(f"{matrices}/{mask_paths[j].stem.removesuffix('_mask')}.dat") # the z-rotation matrix for img

        h, w = img.shape[:2]

        pc_temp = o3d.t.geometry.PointCloud(pc_socs.clone())
        pc_temp.transform(inv(z_rot)).transform(inv(MM)) # Transform the point cloud into the camera's reference frame

        pts = pc_temp.point.positions.numpy() # make the point cloud into something through which I can loop.

        for i, (x, y, z) in enumerate(pts): # loop through each point 
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)): # x, y, and z must be finite
                pc_cmcs.point.px_vals[i,j] = 255
                continue

            if z <= 0: # the point must be in front of the camera
                pc_cmcs.point.px_vals[i,j] = 255
                continue

            if int(pc_temp.point.return_number[i].item()) != 1: # the point must be a single or first return. The is a proxy for occlusion.
                pc_cmcs.point.px_vals[i,j] = 255
                continue

            # calculate c_i = pi(p_i)
            u = ((fx *x) / z) + cx
            v = ((fy *y) / z) + cy

            if not (np.isfinite(u) and np.isfinite(v)): # u and v must be finite
                pc_cmcs.point.px_vals[i,j] = 255
                continue

            # calculate k_i = phi(c_i)
            n = int(np.floor(u + 0.5))
            m = int(np.floor(v + 0.5))

            if 0 <= n < w and 0 <= m < h: # the point must project to the sensor
                pc_cmcs.point.px_vals[i,j] = int(img[m,n])
            else:
                pc_cmcs.point.px_vals[i,j] = 255
                        
    px_vals = pc_cmcs.point.px_vals.numpy() # (N, len(mask_paths)) make the tensor a Numpy array
    class_votes = np.zeros((N, num_classes), dtype=np.int32) # make a (N, num_classes) Numpy array to hold the class_votes

    for c in range(num_classes):
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