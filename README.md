# PointCloud-ML
Collection of scripts for various ML applications

## Modules:
### load_data.py
Loads point clouds as Open3D tensor point clouds, loads transformation matrices, loads mask PNGs. Written for use with classify_from_masks.py

#### load_pngs_from_folder()
Inputs:
- `mask_folder` : Path to the folder containing the mask PNGs

Behavior: 
- For all PNGs in `mask_folder`:
    - read-in the PNG
    - append it to `images`
    - append it's path and name to `valid_paths`

Returns:
- `images` : An array build from the PNGs
- `valid_paths` : An array of paths and file names corresponding to the values in images

#### load_matrices()
Inputs:
- `pop_path` : Path to the folder containing the POP matrix (PRCS>GLCS)
- `sop_path` : Path to the folder containing the SOP matrix (SOCS>PRCS)
- `camera_intrinsics` : Path to the folder containing the camera intrinsics matrix
        - fx, fy
        - cx, cy

Behavior: 
- Read-in each matrix as an array

Returns:
- `POP` : An array containing the POP matrix
- `SOP` : An array containing the SOP matrix
- `intrinsics` : An array containing the camera intrinsics

#### load_point_cloud()
Inputs:
- `pc_folder` : Path to the folder containing the laz point cloud

Behavior:
- As I see it there are 5 "reasonable" scenarios for loading a PC:
    1. las contains [x, y, z] ---> rare but possible
    2. las contains [x, y, z, I] ---> common (I: Intensity)
    3. las contains [x, y, x, R, G, B] ---> common for SFM point clouds 
    4. las contains [x, y, z, I, N] ---> common for a multiple return lidar that lacks a camera (N: Return Number)
    5. las contains [x, y, z, I, R, G, B, N] ---> common for a multiple return lidar with a camera

- I think for now I will only accept scenarios 4 and 5 here, since we require N for occlusion handling and we, in theory, can generate panorama images from the lidar that can be segmented via SAM, so a camera isn't strictly necessary. 

- Read-in the laz point cloud and it's attributes
- Populates arrays with the point cloud attrributes:
    - `points_tensor` : the x, y, and z values for each point
    - `color_tensor` : the red, green, and blue values for each point
    - `intensity_tensor` : the intensity values for each point
- Creates an Open3D tensor point cloud and populates it with the above arrays:
    - `pcd` : the point cloud tensor
        - `pcd.point.positions` : the array contining the positional coordinates
        - `pcd.point.colors` : the array contining the RGB values
        - `pcd.point.intensity` : the array containing the intensity values

Returns:
- `pcd` : An Open3D tensor point cloud in the global reference frame
- `pc_name` : The name of the point cloud

---
### write_data.py
Ingests a classified tensor point cloud and then writes laz to disk. Written for use with classify_from_masks.py

#### write_pcd_to_laz()
Inputs:
- `pcd` : An Open3D tensor point cloud with classification values
- `out_path` : Path to where the output laz file should be written
- `scanpos_name` : The name of the ScanPos for use in naming the output file
- `point_cloud_name` : The name of the input point cloud for use in naming the output file. Returned by `load_point_cloud()`

Behavior: 
- Ingest an Open3D tensor point cloud
    - `pcd` : the point cloud tensor
        - `pcd.point.positions` : the array contining the positional coordinates
        - `pcd.point.colors` : the array contining the RGB values
        - `pcd.point.intensity` : the array containing the intensity values
- Build the LAZ header
    - LAZ version and Point Format
    - Offsets
    - Precision
- Populate the las attributes from the tensor values
    - points
    - colors
    - intensity
    - classification
- Write the laz file to disk

Returns:
- LAZ written to disk

---
### classify_2d_to_3d.py
Ingests a point cloud, 5 matrices, and mask images. Projects the point cloud into mask space and then writes the pixel value to the point cloud. The most common pixel value is kept as the point's classification value. This module was writen for use with classify_from_masks.py.

#### glcs_to_socs()
Inputs:
- `pc_glcs` : An Open3D tensor point cloud in a global reference frame
- `POP` : The matrix transforming points from the PRoject's Coordinate System (PRCS) to the GLobal Coordinate System (GLCS).
- `SOP` : The matrix transforming points from the Scanner's Own Coordinate System (SOCS) to the PRojects Coordinate System (PRCS).

Behavior:
- Ingest an Open3D tensor point cloud in the global reference frame.
- Apply inv(POP) and inv(SOP) to transform the points into the scanner's reference frame

Returns:
- `pc_socs` : An Open3D tensor point cloud in the scanner's reference frame

#### classify_point_cloud()
Inputs:
- `pc_socs` : An Open3D tensor point cloud in the scanner's reference frame
- `pc_glcs` : An Open3D tensor point cloud in a global reference frame
- `masks` : An array of the mask images
- `mask_paths`: An array of paths and file names for each mask image
- `intrinsics` : An array containing the camera intrinsics needed to project the points onto pixels
- `num_classes` : The maximum number of class values contained in the mask images
- `matrices` : The path to the camera mounting and z-rotation matrices to apply to the point cloud to transform the point into the camera's reference frame

Behavior:
- Ingest Point Clouds, matrices, and masks
- For each mask:
    - Translate the points into the camera's reference frame
    - Project the points onto the images
    - Copy the pixel value into `pixel_vals`
- For each class:
    - Count the number of times it shows up in `pixel_vals` and store it in `class_counts`
- For each point:
    - Determine which class appears most often
    - Store that value in `classification`
- Copy `pc_glcs` into a new point cloud, `pc_glcs_classified`
- Add the `classification` array to `pc_glcs_classified`

Returns:
- `pc_glcs_classified` : A copy of the Open3D tensor point cloud with `point.classification` added

------
### interactive_painting_edited.py
Interactive Painting Tools for Multi-Class Image Masking

This module provides interactive tools for painting multi-class masks
on images using OpenCV, designed for 3D semantic labeling workflows.

Tools:
- HDMultiClassMaskPainter: HD interactive masking with large display
- paint_mask_hd: Launch HD masking on a single image
- mask_multiple_images_hd: Batch HD masking across multiple images

Author: Florent Poux
License: learngeodata.eu

Edited to ingest images from TLS scan positions.


## Pipelines:
### classify_from_mask.py
Defines the necessary global variables and paths and calls the various functions from load_data.py, classify_2d_to_3d.py, and write_data.py in the proper order to project the mask values onto the point cloud, chose one singular classification value, and write the classified point cloud to disk as LAZ.

---
### paintMasks.py
Defines the necessary global variables and paths and calls the various functions from interactive_painting_edited.py to open the ScanPos images in the GUI painter, and save the painted masks to disk as PNGs. Heavily adapted from Florent Poux's tutorials.

---

### paintMasksOnly.py
Defines the necessary global variables and paths to save RGB colored masks to disk as PNG when the grayscale masks already exist. Heavily adapted from Florent Poux's tutorials and guided by ChatGPT.


## Not used but have provided inspiration
### da_semantic_masking_edited.py
Tutorial 2: Interactive Semantic Masking & Label Fusion
This tutorial shows how to:
1. Load 3D reconstruction data exported by Tutorial 1
2. Interactively paint HD masks on images
3. Project 2D masks to 3D point clouds
4. Create multi-frame labels for full scenes
5. Propagate labels with smart KD-Tree fusion

Prerequisites:
- Run Tutorial 1 (da_3d_reconstruction.py) first to generate reconstruction_data.npz

Dependencies:
- numpy, open3d, matplotlib, scipy

Author: Florent Poux
License: learngeodata.eu

## Depreciated
### colorize_from_mask.py
The first quasai-pseudocode drafts of what became load_data.py, classify_2d_to_3d.py, write_data.py, and classify_from_mask.py