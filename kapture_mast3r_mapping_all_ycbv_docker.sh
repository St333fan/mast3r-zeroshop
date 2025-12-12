#!/bin/bash

# Bash script to process all objects from 1 to 21 using kapture_mast3r_mapping_all.py
# This script runs the Python script for each object directory

# Base path where all object directories are located
BASE_PATH="/code/datasets/ycbv_real_subset"

# Path to the Python script
PYTHON_SCRIPT="./kapture_mast3r_mapping_all.py"

# Model parameters
MODEL_NAME="MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"

# Check if the Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Check if the base path exists
if [ ! -d "$BASE_PATH" ]; then
    echo "Error: Base path not found at $BASE_PATH"
    exit 1
fi

echo "Starting kapture_mast3r_mapping processing for objects 0 to 21..."
echo "Base path: $BASE_PATH"
echo "Python script: $PYTHON_SCRIPT"
echo "Model name: $MODEL_NAME"
echo "========================================"

# Counter for successful and failed processing
success_count=0
failed_count=0
failed_objects=()

# Loop through objects 1 to 21
for i in {1..21}; do
    # Format the object number with leading zeros (6 digits)
    obj_num=$(printf "%06d" $i)
    obj_path="$BASE_PATH/obj_$obj_num"
    
    echo ""
    echo "Processing object $i (obj_$obj_num)..."
    echo "Object path: $obj_path"
    
    # Check if the object directory exists
    if [ ! -d "$obj_path" ]; then
        echo "Warning: Object directory not found: $obj_path"
        echo "Skipping obj_$obj_num"
        ((failed_count++))
        failed_objects+=("obj_$obj_num (directory not found)")
        continue
    fi
    
    # Check if the required subdirectories exist
    surface_dir="$obj_path/train_pbr/mast3r-sfm/surface/images"
    segmented_dir="$obj_path/train_pbr/mast3r-sfm/segmented/images"
    surface_pairs="$obj_path/train_pbr/mast3r-sfm/surface/images/pairs.txt"
    segmented_pairs="$obj_path/train_pbr/mast3r-sfm/segmented/images/pairs.txt"
    
    if [ ! -d "$surface_dir" ] && [ ! -d "$segmented_dir" ]; then
        echo "Warning: Neither surface nor segmented directory found for obj_$obj_num"
        echo "Expected: $surface_dir or $segmented_dir"
        echo "Skipping obj_$obj_num"
        ((failed_count++))
        failed_objects+=("obj_$obj_num (no valid subdirectories)")
        continue
    fi
    
    # Check if pairs.txt files exist
    if [ ! -f "$surface_pairs" ] && [ ! -f "$segmented_pairs" ]; then
        echo "Warning: No pairs.txt files found for obj_$obj_num"
        echo "Expected: $surface_pairs or $segmented_pairs"
        echo "Skipping obj_$obj_num (run make_pairs_all first)"
        ((failed_count++))
        failed_objects+=("obj_$obj_num (no pairs.txt files)")
        continue
    fi
    
    # Run the Python script for this object
    echo "Running: python3 $PYTHON_SCRIPT --path $obj_path --model_name $MODEL_NAME --use_single_camera"
    
    if python3 "$PYTHON_SCRIPT" --path "$obj_path" --model_name "$MODEL_NAME" --use_single_camera; then
        echo "✓ Successfully processed obj_$obj_num"
        ((success_count++))

        # Activate colmap environment
        source /opt/conda/etc/profile.d/conda.sh
        conda activate colmap

        # Bundler export for both surface and segmented
        for mode in surface segmented; do
            sparse_dir="$obj_path/train_pbr/mast3r-sfm/$mode/sparse/0"
            output_dir="$obj_path/train_pbr/mast3r-sfm/$mode/images/scene"
            if [ -d "$sparse_dir" ]; then
                mkdir -p "$output_dir"
                echo "Exporting bundler for $mode: colmap model_converter --input_path $sparse_dir --output_path $output_dir --output_type BUNDLER"
                if colmap model_converter --input_path "$sparse_dir" --output_path "$output_dir" --output_type BUNDLER; then
                    echo "✓ Bundler export successful for $mode of obj_$obj_num"
                else
                    echo "✗ Bundler export failed for $mode of obj_$obj_num"
                fi
            else
                echo "Warning: $sparse_dir does not exist, skipping bundler export for $mode of obj_$obj_num"
            fi
        done
    else
        echo "✗ Failed to process obj_$obj_num"
        ((failed_count++))
        failed_objects+=("obj_$obj_num")
    fi

        # Return to original environment
        conda activate mast3r

    echo "----------------------------------------"
done

echo ""
echo "========================================"
echo "KAPTURE_MAST3R_MAPPING PROCESSING COMPLETE"
echo "========================================"
echo "Total objects processed: $((success_count + failed_count))"
echo "Successful: $success_count"
echo "Failed: $failed_count"

if [ $failed_count -gt 0 ]; then
    echo ""
    echo "Failed objects:"
    for failed_obj in "${failed_objects[@]}"; do
        echo "  - $failed_obj"
    done
fi

echo ""
if [ $failed_count -eq 0 ]; then
    echo "All objects processed successfully!"
    echo "3D reconstructions have been generated in:"
    echo "  - {object}/train_pbr/mast3r-sfm/surface/reconstruction/"
    echo "  - {object}/train_pbr/mast3r-sfm/segmented/reconstruction/"
    echo "  - {object}/train_pbr/mast3r-sfm/surface/dense/"
    echo "  - {object}/train_pbr/mast3r-sfm/segmented/dense/"
    echo "  - {object}/train_pbr/mast3r-sfm/surface/sparse/"
    echo "  - {object}/train_pbr/mast3r-sfm/segmented/sparse/"
    exit 0
else
    echo "Some objects failed to process. Check the logs above for details."
    exit 1
fi
