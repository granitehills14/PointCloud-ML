import laspy
import cv2
import numpy as np
import colorcet as cc
import matplotlib.colors as mcolors
from pathlib import Path


def define_palette(num_classes):
    if num_classes < 1:
        raise SystemExit("Number of classes must be at least 1... Exiting.")
    
    class0_gray = np.array([[77,77,77]], dtype=np.uint8)
    
    colors = cc.glasbey[:num_classes]

    colors_rgb = np.array(
        [
            np.array(mcolors.to_rgb(color)) * 255
            for color in colors
        ],
        dtype=np.uint8
    )

    palette = np.vstack([class0_gray, colors_rgb])

    return palette

def write_pcd_to_laz(pcd, out_path, scanpos_name, point_cloud_name):
    precision = 0.00025 # 0.00025m for Riegl TLS, UAS, and MLS instruments

    points = pcd.point.positions.numpy() # point x, y, z values

    has_rgb = 'colors' in pcd.point
    has_intensity = 'intensity' in pcd.point
    has_class = 'classification' in pcd.point
    has_return_number = 'return_number' in pcd.point

    PF = 7 if has_rgb else 6
    
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


def write_mask(mask, name, output_path, color, palette):
    '''
    Writes the combined mask to the disk. 
    '''

    out_path = Path(f"{output_path}/MASKS/")
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = Path(f"{out_path}/{name.stem}_mask.png")
                    
    cv2.imwrite(str(out_file), mask)
    print(f"Saved {out_file}")

    if color:
        color_path = Path(f"{output_path}/MASKS/COLOR/")
        color_path.mkdir(parents=True, exist_ok=True)
        colored_mask = palette[mask]
        colored_file = Path(f"{color_path}/{name.stem}_mask_colored.png")
        cv2.imwrite(str(colored_file), colored_mask[:,:,::-1])
        print(f"Saved {colored_file}")

def write_confidence(confidence, name, output_path):
    '''
    Writes the combined confidence to the disk. 
    '''

    out_path = Path(f"{output_path}/MASKS/")
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = Path(f"{out_path}/{name.stem}_confidence.png")
                    
    cv2.imwrite(str(out_file), confidence)
    print(f"Saved {out_file}")