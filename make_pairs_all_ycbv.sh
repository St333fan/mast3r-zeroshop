#!/bin/bash

# Bash script to process all objects from 0 to 21 using make_pairs_all.py
# This script runs the Python script for each object directory

# Base path where all object directories are located
BASE_PATH="/home/st3fan/Projects/Grounded-SAM-2/dataset/test"

# Path to the Python script
PYTHON_SCRIPT="make_pairs_all.py"

# Model parameters
MODEL_NAME="MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
RETRIEVAL_MODEL="./checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"

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

# Check if the retrieval model exists
if [ ! -f "$RETRIEVAL_MODEL" ]; then
    echo "Error: Retrieval model not found at $RETRIEVAL_MODEL"
    exit 1
fi

echo "Starting make_pairs processing for objects 0 to 21..."
echo "Base path: $BASE_PATH"
echo "Python script: $PYTHON_SCRIPT"
echo "Model name: $MODEL_NAME"
echo "Retrieval model: $RETRIEVAL_MODEL"
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
    
    if [ ! -d "$surface_dir" ] && [ ! -d "$segmented_dir" ]; then
        echo "Warning: Neither surface nor segmented directory found for obj_$obj_num"
        echo "Expected: $surface_dir or $segmented_dir"
        echo "Skipping obj_$obj_num"
        ((failed_count++))
        failed_objects+=("obj_$obj_num (no valid subdirectories)")
        continue
    fi
    
    # Run the Python script for this object
    echo "Running: python3 $PYTHON_SCRIPT --path $obj_path --model_name $MODEL_NAME --retrieval_model $RETRIEVAL_MODEL"
    
    if python3 "$PYTHON_SCRIPT" --path "$obj_path" --model_name "$MODEL_NAME" --retrieval_model "$RETRIEVAL_MODEL"; then
        echo "✓ Successfully processed obj_$obj_num"
        ((success_count++))
    else
        echo "✗ Failed to process obj_$obj_num"
        ((failed_count++))
        failed_objects+=("obj_$obj_num")
    fi
    
    echo "----------------------------------------"
done

echo ""
echo "========================================"
echo "MAKE_PAIRS PROCESSING COMPLETE"
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
    echo "Pairs files have been generated in:"
    echo "  - {object}/train_pbr/mast3r-sfm/surface/images/pairs.txt"
    echo "  - {object}/train_pbr/mast3r-sfm/segmented/images/pairs.txt"
    exit 0
else
    echo "Some objects failed to process. Check the logs above for details."
    exit 1
fi
