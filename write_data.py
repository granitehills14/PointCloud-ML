import os
import numpy as np
import laspy
import open3d as o3d

def write_pcd_to_laz(pcd, out_path, scanpos_name, point_cloud_name):
    precision = 0.00025 # 0.00025m for Riegl TLS, UAS, and MLS instruments

    points = pcd.point.positions.numpy() # point x, y, z values
    
    if 'colors' in pcd.point:
        colors = pcd.point.colors.numpy() * 65535 # point RGB values
    
    if 'intensity' in pcd.point:
        intensity = pcd.point.intensity.numpy() # point intensity values
    
    if 'classification' in pcd.point:
        classification = pcd.point.classification.numpy() # point classification values (from classify_point_cloud())
    
    header = laspy.LasHeader(point_format=6, version="1.4") # set-up header LAS 1.4 PF:6, change if needed (can make these variables if I want)
    header.offsets = np.min(points, axis=0)
    header.scales = [precision, precision, precision]

    las = laspy.LasData(header)

    las.x = points[:,0]
    las.y = points[:,1]
    las.z = points[:,2]

    las.red = colors[:, 0]
    las.green = colors[:, 1]
    las.blue = colors[:, 2]

    las.intensity = intensity.flatten().astype(np.uint16)

    las.classification = classification.flatten().astype(np.uint8)

    las.write(f"{out_path}/{scanpos_name}_{point_cloud_name}.laz")
    print(f"Writing {scanpos_name}_{point_cloud_name}.laz to {out_path}")