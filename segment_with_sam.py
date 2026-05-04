'''
segment_with_sam.py

Scaffold for generating masks for a folder of images given text prompts.
'''

from PIL import Image
import requests
from io import BytesIO
import torch
import sys
import os
import numpy as np
from typing import List
from transformers import Sam3Processor, Sam3Model
import load_data as ld
import write_data as wd

# Choose device: CUDA (NVIDIA GPU), CPU (all others)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define paths and parameters
config_path = "./"
config_json = "config.json"
settings, sam3, num_classes = ld.load_config(config_path, config_json)
MODEL = sam3["model"]
prompts = sam3["prompts"]

SCENE = settings["SCENE"]
SCANPOS = settings["SCANPOS"]
project_dir = f"./{SCENE}"
scanpos_dir = f"./{SCENE}/{SCANPOS}"

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

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


# Initialize SAM
model = Sam3Model.from_pretrained(MODEL).to(device)
processor = Sam3Processor.from_pretrained(MODEL)

images, valid_paths = ld.load_raw_imagery(paths['raw'])
text_prompts = sam3["prompts"]

'''
For every image in the image folder:
    Create empty mask array
    Create empty confidence array
    Pre-compute image embeddings

    For every prompt in prompts:
        Create a class_id
        Set the prompt
        Run SAM3
        Post-process the SAM3 outputs
            Determine if there are cells to change
            Determine which cells need updating
            Update the cells
            Update the confidence
    
    Write the mask and confidence to disk as PNG

'''
for image, image_path in zip(images, valid_paths):
    combined_mask = np.zeros((image.shape[:2])) # create empty mask array
    mask_confidence = np.zeros((image.shape[:2])) # create empty mask confidence array
    
    # Pre-compute image embeddings (per HF Transformers Docs)
    img_inputs = processor(
        images=image,
        return_tensors="pt"
        ).to(device)
    
    with torch.no_grad():
        vision_embeds = model.get_vision_features(
            pixel_values=img_inputs.pixel_values
            )

    for prompt in text_prompts:
        class_id = text_prompts.index(prompt) + 1 # create class id

       # set text prompt 
        text_inputs = processor(
            text=prompt,
            return_tensors="pt"
        ).to(device)

        # run SAM3
        with torch.no_grad():
            outputs = model(
                vision_embeds=vision_embeds, 
                **text_inputs
                )
        
        # Post-process results
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=img_inputs.get("original_sizes").tolist()
        )[0]

        masks = to_numpy(results["masks"])
        
        if len(masks) == 0:
            continue

        confidence = to_numpy(results["scores"])

        masks_bool = masks.astype(bool)

        confidence_per_mask = confidence[:, None, None]

        combined_confidence = np.where(
            masks_bool,
            confidence_per_mask,
            0
        )

        prompt_confidence = combined_confidence.max(axis=0)


        # assess only the mask for this prompt
        prompt_mask = masks.any(axis=0)

        # is there a mask value?
        candidate_exists = (prompt_mask == 1) 
        
        # which pixels should be updated?
        update_pixels = candidate_exists & (prompt_confidence > mask_confidence)

        # update the combined mask array
        combined_mask[update_pixels] = class_id

        # update the combined confidence array
        mask_confidence[update_pixels] = prompt_confidence[update_pixels]

    # write the mask and confidence to disk
    wd.write_mask(combined_mask, image_path, paths['out'])
    wd.write_confidence(mask_confidence, image_path, paths['out'])