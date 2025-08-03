#!/usr/bin/env python3
"""
Script to prepare directory structure for mast3r processing.
Generates rgb_mask folders and distributes images to appropriate directories.
"""

import os
import cv2
import numpy as np
from PIL import Image
import argparse
import shutil


def combine_rgb_mask(rgb_folder, mask_folder, output_folder):
    """
    Combines RGB images with their corresponding masks to create RGBA images.
    Handles the naming pattern: RGB: 000000.png, Mask: 000000_000000.png
    
    Args:
        rgb_folder (str): Path to RGB images folder
        mask_folder (str): Path to mask images folder
        output_folder (str): Path to output folder for RGBA images
    """
    # Check if input folders exist
    if not os.path.exists(rgb_folder):
        print(f"Error: RGB folder not found at {rgb_folder}")
        return False
    
    if not os.path.exists(mask_folder):
        print(f"Error: Mask folder not found at {mask_folder}")
        return False
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get list of RGB images
    rgb_files = [f for f in os.listdir(rgb_folder) if f.lower().endswith('.png')]
    rgb_files.sort()  # Sort to ensure consistent processing order
    
    if not rgb_files:
        print("No PNG files found in RGB folder")
        return False
    
    processed_count = 0
    
    for rgb_file in rgb_files:
        # Get the base filename without extension (e.g., "000000" from "000000.png")
        base_name = os.path.splitext(rgb_file)[0]
        
        # Construct the corresponding mask filename (e.g., "000000_000000.png")
        mask_file = f"{base_name}_000000.png"
        mask_path = os.path.join(mask_folder, mask_file)
        
        if not os.path.exists(mask_path):
            print(f"Warning: No corresponding mask found for {rgb_file} (looking for {mask_file})")
            continue
        
        try:
            # Load RGB image
            rgb_path = os.path.join(rgb_folder, rgb_file)
            rgb_image = cv2.imread(rgb_path)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            
            # Load mask image
            mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Ensure both images have the same dimensions
            if rgb_image.shape[:2] != mask_image.shape:
                print(f"Warning: Size mismatch for {rgb_file}. Resizing mask to match RGB image.")
                mask_image = cv2.resize(mask_image, (rgb_image.shape[1], rgb_image.shape[0]))
            
            # Create RGBA image
            rgba_image = np.zeros((rgb_image.shape[0], rgb_image.shape[1], 4), dtype=np.uint8)
            rgba_image[:, :, :3] = rgb_image  # RGB channels
            rgba_image[:, :, 3] = mask_image  # Alpha channel
            
            # Convert to PIL Image and save as PNG
            pil_image = Image.fromarray(rgba_image, 'RGBA')
            output_path = os.path.join(output_folder, f"{base_name}.png")
            pil_image.save(output_path)
            
            processed_count += 1
            print(f"Processed: {rgb_file} + {mask_file} -> {base_name}.png")
            
        except Exception as e:
            print(f"Error processing {rgb_file}: {str(e)}")
            continue
    
    print(f"Completed! Processed {processed_count} images in {output_folder}")
    return processed_count > 0


def copy_images_to_directory(source_folder, dest_folder, image_range, image_type="rgb_mask"):
    """
    Copy specific range of images to destination folder.
    
    Args:
        source_folder (str): Source folder containing images
        dest_folder (str): Destination folder
        image_range (tuple): Range of images to copy (start, end) - end is exclusive
        image_type (str): Type of images being copied (for logging)
    """
    os.makedirs(dest_folder, exist_ok=True)
    
    start_idx, end_idx = image_range
    copied_count = 0
    
    for i in range(start_idx, end_idx):
        source_file = os.path.join(source_folder, f"{i:06d}.png")
        dest_file = os.path.join(dest_folder, f"{i:06d}.png")
        
        if os.path.exists(source_file):
            shutil.copy2(source_file, dest_file)
            copied_count += 1
        else:
            print(f"Warning: Source file not found: {source_file}")
    
    print(f"Copied {copied_count} {image_type} images ({start_idx:06d}-{end_idx-1:06d}) to {dest_folder}")


def copy_and_rename_mask_images(source_folder, dest_folder, image_range):
    """
    Copy specific range of mask images to destination folder and rename them.
    Removes the _000000 suffix from filenames.
    
    Args:
        source_folder (str): Source folder containing mask images
        dest_folder (str): Destination folder
        image_range (tuple): Range of images to copy (start, end) - end is exclusive
    """
    os.makedirs(dest_folder, exist_ok=True)
    
    start_idx, end_idx = image_range
    copied_count = 0
    
    for i in range(start_idx, end_idx):
        source_file = os.path.join(source_folder, f"{i:06d}_000000.png")
        dest_file = os.path.join(dest_folder, f"{i:06d}.png")  # Remove _000000 suffix
        
        if os.path.exists(source_file):
            shutil.copy2(source_file, dest_file)
            copied_count += 1
        else:
            print(f"Warning: Source mask file not found: {source_file}")
    
    print(f"Copied and renamed {copied_count} mask images ({start_idx:06d}-{end_idx-1:06d}) to {dest_folder}")


def copy_mask_images_to_directory(source_folder, dest_folder, image_range):
    """
    Copy specific range of mask images to destination folder.
    
    Args:
        source_folder (str): Source folder containing mask images
        dest_folder (str): Destination folder
        image_range (tuple): Range of images to copy (start, end) - end is exclusive
    """
    os.makedirs(dest_folder, exist_ok=True)
    
    start_idx, end_idx = image_range
    copied_count = 0
    
    for i in range(start_idx, end_idx):
        source_file = os.path.join(source_folder, f"{i:06d}_000000.png")
        dest_file = os.path.join(dest_folder, f"{i:06d}_000000.png")
        
        if os.path.exists(source_file):
            shutil.copy2(source_file, dest_file)
            copied_count += 1
        else:
            print(f"Warning: Source mask file not found: {source_file}")
    
    print(f"Copied {copied_count} mask images ({start_idx:06d}-{end_idx-1:06d}) to {dest_folder}")


def postprocess_segmented_images(segmented_images_folder):
    """
    Post-process segmented RGBA images to set background pixels to black.
    Uses the alpha channel of the RGBA images to identify background areas.
    
    Args:
        segmented_images_folder (str): Path to segmented images folder containing RGBA images
    """
    print(f"\nPost-processing segmented images in: {segmented_images_folder}")
    
    if not os.path.exists(segmented_images_folder):
        print(f"Error: Segmented images folder not found: {segmented_images_folder}")
        return False
    
    # Get list of RGBA images
    rgba_files = [f for f in os.listdir(segmented_images_folder) if f.lower().endswith('.png')]
    rgba_files.sort()
    
    if not rgba_files:
        print("No PNG files found in segmented images folder")
        return False
    
    processed_count = 0
    
    for rgba_file in rgba_files:
        try:
            # Load RGBA image
            rgba_path = os.path.join(segmented_images_folder, rgba_file)
            rgba_image = cv2.imread(rgba_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel
            
            if rgba_image.shape[2] != 4:
                print(f"Warning: {rgba_file} is not RGBA format, skipping...")
                continue
            
            # Use alpha channel to create background mask
            alpha_channel = rgba_image[:, :, 3]
            background_mask = alpha_channel < 10  # Threshold for background detection
            
            # Set RGB channels to black for background pixels, keep alpha unchanged
            rgba_image[background_mask, 0] = 0  # Blue channel (OpenCV uses BGR)
            rgba_image[background_mask, 1] = 0  # Green channel
            rgba_image[background_mask, 2] = 0  # Red channel
            # Alpha channel (index 3) remains unchanged
            
            # Save the modified RGBA image
            cv2.imwrite(rgba_path, rgba_image)
            
            processed_count += 1
            print(f"Post-processed: {rgba_file} (set background to black using alpha channel)")
            
        except Exception as e:
            print(f"Error post-processing {rgba_file}: {str(e)}")
            continue
    
    print(f"Post-processing completed! Modified {processed_count} segmented images.")
    return processed_count > 0


def process_object_directory(obj_path):
    """
    Process a single object directory to create rgb_mask folders and distribute images.
    
    Args:
        obj_path (str): Path to the object directory (e.g., "obj_000003")
    """
    print(f"\nProcessing object directory: {obj_path}")
    
    # Define the base paths
    train_pbr_path = os.path.join(obj_path, "train_pbr")
    base_000000_path = os.path.join(train_pbr_path, "000000")
    mast3r_sfm_path = os.path.join(train_pbr_path, "mast3r-sfm")
    
    # Check if the required directories exist
    if not os.path.exists(base_000000_path):
        print(f"Error: Base directory not found: {base_000000_path}")
        return False
    
    # Define source folders
    rgb_folder = os.path.join(base_000000_path, "rgb")
    mask_folder = os.path.join(base_000000_path, "mask")
    
    # Create rgb_mask folder in 000000 (generate once)
    rgb_mask_000000 = os.path.join(base_000000_path, "rgb_mask")
    print(f"Creating rgb_mask in: {rgb_mask_000000}")
    if not combine_rgb_mask(rgb_folder, mask_folder, rgb_mask_000000):
        print(f"Failed to create rgb_mask in {rgb_mask_000000}")
        return False
    
    # Create necessary mast3r-sfm directory structure
    segmented_path = os.path.join(mast3r_sfm_path, "segmented")
    surface_path = os.path.join(mast3r_sfm_path, "surface")
    
    segmented_images = os.path.join(segmented_path, "images")
    segmented_masks = os.path.join(segmented_path, "masks")
    surface_images = os.path.join(surface_path, "images")
    surface_masks = os.path.join(surface_path, "masks")
    
    # Copy and rename original mask files to masks folders (not rgb_mask images)
    print(f"Copying original mask files to: {segmented_masks}")
    copy_and_rename_mask_images(mask_folder, segmented_masks, (0, 30))
    
    print(f"Copying original mask files to: {surface_masks}")
    copy_and_rename_mask_images(mask_folder, surface_masks, (0, 20))
    
    # Distribute rgb_mask images according to specifications
    # Surface: images 0-19 (20 images)
    print("\nDistributing surface images...")
    copy_images_to_directory(rgb_mask_000000, surface_images, (0, 20), "rgb_mask")
    
    # Segmented: images 0-29 (30 images)  
    print("\nDistributing segmented images...")
    copy_images_to_directory(rgb_mask_000000, segmented_images, (0, 30), "rgb_mask")
    
    # Post-process segmented images to set background to black
    postprocess_segmented_images(segmented_images)
    
    print(f"Successfully processed: {obj_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Prepare directory structure for mast3r processing')
    parser.add_argument('obj_path', help='Path to object directory (e.g., "obj_000003")')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.obj_path):
        print(f"Error: Object path {args.obj_path} does not exist")
        return
    
    if not os.path.isdir(args.obj_path):
        print(f"Error: {args.obj_path} is not a directory")
        return
    
    success = process_object_directory(args.obj_path)
    
    if success:
        print(f"\n✓ Successfully processed {args.obj_path}")
        print("\nGenerated structure:")
        print(f"  - {args.obj_path}/train_pbr/000000/rgb_mask (all 30 images)")
        print(f"  - {args.obj_path}/train_pbr/mast3r-sfm/segmented/masks (30 renamed mask images)")
        print(f"  - {args.obj_path}/train_pbr/mast3r-sfm/segmented/images (0-29 rgb_mask images with black backgrounds)")
        print(f"  - {args.obj_path}/train_pbr/mast3r-sfm/surface/masks (20 renamed mask images)")
        print(f"  - {args.obj_path}/train_pbr/mast3r-sfm/surface/images (0-19 rgb_mask images)")
    else:
        print(f"\n✗ Failed to process {args.obj_path}")


if __name__ == "__main__":
    main()