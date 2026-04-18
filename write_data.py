import numpy as np
import laspy

def write_pcd_to_laz(pcd, out_path, scanpos_name, point_cloud_name):
    precision = 0.00025 # 0.00025m for Riegl TLS, UAS, and MLS instruments

    points = pcd.point.positions.numpy() # point x, y, z values

    has_rgb = 'colors' in pcd.point
    has_intensity = 'intensity' in pcd.point
    has_class = 'classification' in pcd.point
    has_return_number = 'return_number' in pcd.point

    if has_rgb:
        PF = 7
    else: 
        PF = 6
    
    header = laspy.LasHeader(point_format=PF, version="1.4") # set-up header LAS 1.4 PF:7 if we have rgb, 6 otherwise.
    header.offsets = np.min(points, axis=0)
    header.scales = [precision, precision, precision]

    las = laspy.LasData(header)

    las.x = points[:,0]
    las.y = points[:,1]
    las.z = points[:,2]

    if has_rgb:
        colors = (pcd.point.colors.numpy() * 65535).astype(np.uint16) # point RGB values
        las.red = colors[:, 0]
        las.green = colors[:, 1]
        las.blue = colors[:, 2]

    if has_intensity:
        intensity = pcd.point.intensity.numpy() # point intensity values
        las.intensity = intensity.flatten().astype(np.uint16)
    
    if has_class:
        classification = pcd.point.classification.numpy() # point classification values (from classify_point_cloud())
        las.classification = classification.flatten().astype(np.uint8)
    
    if has_return_number:
        Return_Number = pcd.point.return_number.numpy()
        las.return_number = Return_Number.flatten().astype(np.uint8)
        
    las.write(f"{out_path}/{scanpos_name}_{point_cloud_name}.laz")
    print(f"Writing {scanpos_name}_{point_cloud_name}.laz to {out_path}")