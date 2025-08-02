#!/bin/bash

# Bash script to process all objects from 0 to 21 using prepare_structure.py
# This script runs the Python script for each object directory

# Base path where all object directories are located
BASE_PATH="/home/stefan/Downloads/objs_sizex10/objs_texture_sizex10"

# Path to the Python script
PYTHON_SCRIPT="/home/stefan/PycharmProjects/mast3r/prepare_structure.py"

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

echo "Starting processing of objects 0 to 21..."
echo "Base path: $BASE_PATH"
echo "Python script: $PYTHON_SCRIPT"
echo "========================================"

# Counter for successful and failed processing
success_count=0
failed_count=0
failed_objects=()

# Loop through objects 0 to 21
for i in {0..21}; do
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
    
    # Run the Python script for this object
    echo "Running: python3 $PYTHON_SCRIPT $obj_path"
    
    if python3 "$PYTHON_SCRIPT" "$obj_path"; then
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
echo "PROCESSING COMPLETE"
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
    echo "🎉 All objects processed successfully!"
    exit 0
else
    echo "⚠️  Some objects failed to process. Check the logs above for details."
    exit 1
fi
