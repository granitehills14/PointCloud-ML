'''
segment_with_sam.py

Scaffold for generating masks for a folder of images given text prompts.
'''

from PIL import Image
import requests
from io import BytesIO
import sam3
from sam3.model_builder import build_sam3_image_model
# from sam3.model.sam3_image_processor import Sam3Processor
from sam3.train.data.collator import collate_fn_api as collate
from sam3.model.utils.misc import copy_data_to_device
import os
sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
import torch
import sys
sys.path.append(f"{sam3_root}/examples")
from sam3.visualization_utils import plot_results
from sam3.train.data.sam3_image_dataset import InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint
from typing import List
from sam3 import build_sam3_image_model
from transformers import Sam3Processor, Sam3Model
import load_data as ld
import write_data as wd

# Choose device: CUDA (NVIDIA GPU), CPU (all others)
device = "cuda" if torch.cuda.is_available() else "cpu"
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

# Initialize SAM
bpe_path = f"{sam3_root}/assets/bpe_sample_vocab_16e6.txt.gz"
model = Sam3Model.from_pretrained(MODEL)
processor = Sam3Processor(MODEL)

images, valid_paths = ld.load_raw_images(paths['raw'])

'''
for every image in images:
    for every text query:
        run inference using SAM
        make the boolean mask into a real semantic mask
'''

for image in images: # this code is pretty trash
    counter = 0
    for i in range(num_classes):
    
        inputs = processor(
            images=datapoints[i].images[i],
            text=datapoints[i].find_queries[i].astype('str')
        )

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )
        
        if output["masks"].shape[0] > 0:
            prompt_mask = outputs["masks"][:,0]
            combined_masks[prompt_mask.cpu().numpy()] = counter
        
        counter += 1

'''
    for every mask image from the previous loop:
        if cell is not empty:
            what is the cell value?
            what mask does that map to?
            if confidence of existing value > confidence of new value:
                do nothing
            else:
                combined_mask_cell = mask_cell
'''

    wd.write_mask_to_disk(combined_mask)





'''fig = plt.figure(figsize=(np.shape(image_rgb)[1]/72, np.shape(image_rgb)[0]/72))
fig.add_axes([0,0,1,1])
plt.imshow(image_rgb)
color_mask = sam_masks(result)
plt.axes('off')
plt.savefig("../test_result.jpeg")'''