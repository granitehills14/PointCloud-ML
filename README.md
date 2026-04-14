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


### write_data.py
Ingests a classified tensor point cloud and then writes laz to disk. Written for use with classify_from_masks.py

#### write_pcd_to_laz()
Inputs:
- `pcd` : A Open3D tensor point cloud with classification values
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