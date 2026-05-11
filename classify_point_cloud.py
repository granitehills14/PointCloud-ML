'''
Dominic Filiano
classify_point_cloud.py

Chains the SAM3 classification step together with the point projection and segmentation transfer step for a single, unified script.
'''

if __name__ == "__main__":
    #%% Step 0: Import Dependencies
    import os
    import torch
    import numpy as np
    from transformers import Sam3Processor, Sam3Model
    import load_data as ld
    import write_data as wd
    from classify_2d_to_3d import glcs_to_socs, classify_point_cloud

    #%% Step 1: Establish Script-Wide Variables and Paths
    config_path = "./pcml/code/"
    config_json = "config.json"
    settings, sam3, num_classes = ld.load_config(config_path, config_json)
    MODEL = sam3["model"]
    prompts = sam3["prompts"]
    color = settings["color_masks"]

    SCENE = settings["SCENE"]
    SCANPOS = settings["SCANPOS"]
    project_dir = f"./pcml/data/riegl/{SCENE}"
    scanpos_dir = f"./pcml/data/riegl/{SCENE}/{SCANPOS}"

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

    # if CUDA is available, use it, otherwise use the CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    # If wanting colored masks, define color palette 
    palette = wd.define_palette(num_classes) if color else None

    #%% Step 2: Perform Image Segmentation using SAM3
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
    # Initialize SAM
    model = Sam3Model.from_pretrained(MODEL).to(device)
    processor = Sam3Processor.from_pretrained(MODEL)

    # Load images
    images, valid_paths = ld.load_raw_imagery(paths['raw'])

    # Perform segmentation
    for image, image_path in zip(images, valid_paths):
        print(f"Performing inference on {image_path}")
        combined_mask = np.zeros((image.shape[:2]), dtype=np.uint8) # create empty mask array
        mask_confidence = np.zeros((image.shape[:2]), dtype=np.float32) # create empty mask confidence array
        
        # Pre-compute image embeddings (per HF Transformers Docs)
        img_inputs = processor(
            images=image,
            return_tensors="pt"
            ).to(device)
        
        with torch.no_grad():
            vision_embeds = model.get_vision_features(
                pixel_values=img_inputs.pixel_values
                )

        for class_id, prompt in enumerate(prompts, start=1):
            print(f"Computing masks for {prompt}")

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

            # is there a mask value?
            candidate_exists = masks_bool.any(axis=0) 
            
            # which pixels should be updated?
            update_pixels = candidate_exists & (prompt_confidence > mask_confidence)

            # update the combined mask array
            combined_mask[update_pixels] = class_id

            # update the combined confidence array
            mask_confidence[update_pixels] = prompt_confidence[update_pixels]

        # write the mask and confidence to disk
        wd.write_mask(combined_mask, image_path, paths['masks'], color, palette)
        wd.write_confidence(mask_confidence, image_path, SCANPOS, paths['out'])

    #%% Step 3: Perform Point Projection and Segmentation Transfer
    # Load matrices
    POP, SOP, intrinsics = ld.load_matrices(paths['pop'], paths['sop'], paths['intrinsics'])

    # Load point cloud
    pc_glcs, pc_name = ld.load_point_cloud(paths['data'])

    # Load Masks

    masks, mask_paths = ld.load_pngs_from_folder(paths['masks'])

    # Transfer classification from masks to point cloud
    pc_socs = glcs_to_socs(pc_glcs, POP, SOP)
    pc_glcs_classified = classify_point_cloud(pc_socs, pc_glcs, masks, mask_paths, intrinsics, num_classes, paths['matrices'])

    # Write classified point cloud to disk
    wd.write_pcd_to_laz(pc_glcs_classified, paths['out'], SCANPOS, pc_name)