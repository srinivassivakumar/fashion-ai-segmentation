#!/usr/bin/env python3
"""
========================================================================================================
POST-PROCESSING SCRIPT: CENTER AND CROP CLOTHING ITEMS
========================================================================================================

This script processes the clothing crops from SegFormer+SAM extraction and:
1. Removes excess white background
2. Finds the actual bounding box of the clothing item
3. Crops to the clothing boundaries
4. Centers the item with configurable padding
5. Auto-sizes to fit the clothing dimensions

INPUT:  outputs/crops/*.png (original crops with lots of white space)
OUTPUT: outputs/crops_centered/*.png (centered crops with minimal padding)

Author: AI-Assisted Fashion Segmentation System
Date: 2025
========================================================================================================
"""

import os
import cv2
import numpy as np
from PIL import Image
import glob
from pathlib import Path

# ========================================================================================================
# CONFIGURATION
# ========================================================================================================

class Config:
    """Configuration for post-processing"""
    
    # Paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "crops")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "crops_centered")
    
    # Processing parameters
    PADDING = 50  # Padding around the clothing item (pixels)
    BACKGROUND_COLOR = (255, 255, 255)  # White background
    WHITE_THRESHOLD = 250  # Pixels above this value are considered background
    
    # Output options
    MAINTAIN_ASPECT_RATIO = True
    AUTO_SIZE = True  # Fit to clothing dimensions (not fixed size)

# ========================================================================================================
# CORE FUNCTIONS
# ========================================================================================================

def find_clothing_bbox(image_rgb, white_threshold=250):
    """
    Find the bounding box of the actual clothing item (non-white pixels)
    
    Args:
        image_rgb: RGB image (numpy array)
        white_threshold: Pixel values above this are considered background
    
    Returns:
        (x, y, w, h): Bounding box coordinates, or None if no clothing found
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # Find non-white pixels (the clothing)
    mask = gray < white_threshold
    
    # Find coordinates of non-white pixels
    coords = np.argwhere(mask)
    
    if len(coords) == 0:
        print("[WARN] No clothing pixels found (all white)")
        return None
    
    # Get bounding box
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    x = x_min
    y = y_min
    w = x_max - x_min + 1
    h = y_max - y_min + 1
    
    return (x, y, w, h)

def center_and_crop_clothing(image_rgb, padding=50, background_color=(255, 255, 255)):
    """
    Center the clothing item with minimal padding
    
    Args:
        image_rgb: Original image with clothing on white background
        padding: Pixels of padding around the clothing
        background_color: Background color (default white)
    
    Returns:
        centered_image: Cropped and centered image
        bbox: Original bounding box for reference
    """
    # Find the clothing bounding box
    bbox = find_clothing_bbox(image_rgb, white_threshold=Config.WHITE_THRESHOLD)
    
    if bbox is None:
        print("[ERROR] Could not find clothing in image")
        return None, None
    
    x, y, w, h = bbox
    
    print(f"  [INFO] Found clothing: bbox=[{x}, {y}, {w}, {h}]")
    
    # Crop to the bounding box with padding
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image_rgb.shape[1], x + w + padding)
    y2 = min(image_rgb.shape[0], y + h + padding)
    
    # Crop the image
    cropped = image_rgb[y1:y2, x1:x2]
    
    # Create new canvas with padding
    new_h, new_w = cropped.shape[:2]
    
    # Add extra padding to create final canvas
    canvas_h = new_h + 2 * padding
    canvas_w = new_w + 2 * padding
    
    # Create white canvas
    canvas = np.full((canvas_h, canvas_w, 3), background_color, dtype=np.uint8)
    
    # Calculate center position
    y_offset = padding
    x_offset = padding
    
    # Place cropped image on canvas
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = cropped
    
    print(f"  [INFO] Output size: {canvas_w}x{canvas_h} (clothing: {w}x{h}, padding: {padding}px)")
    
    return canvas, bbox

def create_tight_crop(image_rgb, padding=50):
    """
    Create a tight crop around the clothing with minimal white space
    This is the simplest approach - just crop to bbox + padding
    
    Args:
        image_rgb: Original image
        padding: Padding around clothing
    
    Returns:
        Tightly cropped image
    """
    bbox = find_clothing_bbox(image_rgb, white_threshold=Config.WHITE_THRESHOLD)
    
    if bbox is None:
        return image_rgb  # Return original if nothing found
    
    x, y, w, h = bbox
    
    # Calculate crop boundaries with padding
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image_rgb.shape[1], x + w + padding)
    y2 = min(image_rgb.shape[0], y + h + padding)
    
    # Crop
    cropped = image_rgb[y1:y2, x1:x2]
    
    print(f"  [INFO] Tight crop: {cropped.shape[1]}x{cropped.shape[0]} (original clothing: {w}x{h})")
    
    return cropped

# ========================================================================================================
# BATCH PROCESSING
# ========================================================================================================

def process_single_crop(input_path, output_dir, method='tight'):
    """
    Process a single crop image
    
    Args:
        input_path: Path to input crop image
        output_dir: Directory to save output
        method: 'tight' for tight crop, 'centered' for centered with padding
    
    Returns:
        True if successful, False otherwise
    """
    filename = os.path.basename(input_path)
    print(f"\n[PROCESSING] {filename}")
    
    # Load image
    try:
        image = Image.open(input_path)
        image_rgb = np.array(image.convert('RGB'))
    except Exception as e:
        print(f"  [ERROR] Failed to load image: {e}")
        return False
    
    original_size = image_rgb.shape[:2]
    print(f"  [INFO] Original size: {image_rgb.shape[1]}x{image_rgb.shape[0]}")
    
    # Process based on method
    if method == 'tight':
        result = create_tight_crop(image_rgb, padding=Config.PADDING)
    else:  # centered
        result, bbox = center_and_crop_clothing(
            image_rgb, 
            padding=Config.PADDING,
            background_color=Config.BACKGROUND_COLOR
        )
    
    if result is None:
        print(f"  [ERROR] Failed to process image")
        return False
    
    # Calculate size reduction
    original_pixels = original_size[0] * original_size[1]
    new_pixels = result.shape[0] * result.shape[1]
    reduction = (1 - new_pixels / original_pixels) * 100
    
    print(f"  [INFO] Size reduction: {reduction:.1f}%")
    
    # Save result
    output_path = os.path.join(output_dir, filename)
    try:
        Image.fromarray(result).save(output_path)
        print(f"  [OK] Saved: {output_path}")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to save: {e}")
        return False

def batch_process_crops(method='tight'):
    """
    Process all crop images in the input directory
    
    Args:
        method: 'tight' for tight crop, 'centered' for centered with padding
    """
    print("=" * 80)
    print("POST-PROCESSING: CENTER AND CROP CLOTHING ITEMS")
    print("=" * 80)
    print(f"Input directory: {Config.INPUT_DIR}")
    print(f"Output directory: {Config.OUTPUT_DIR}")
    print(f"Method: {method}")
    print(f"Padding: {Config.PADDING}px")
    print(f"White threshold: {Config.WHITE_THRESHOLD}")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    print(f"\n[OK] Output directory created: {Config.OUTPUT_DIR}")
    
    # Find all crop images
    image_patterns = ['*.png', '*.jpg', '*.jpeg']
    image_files = []
    
    for pattern in image_patterns:
        image_files.extend(glob.glob(os.path.join(Config.INPUT_DIR, pattern)))
    
    if not image_files:
        print(f"\n[ERROR] No images found in {Config.INPUT_DIR}")
        print(f"[HINT] Run segformer_sam_extractor.py first to generate crops")
        return
    
    print(f"\n[INFO] Found {len(image_files)} images to process\n")
    
    # Process each image
    successful = 0
    failed = 0
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n{'=' * 80}")
        print(f"IMAGE {i}/{len(image_files)}")
        print(f"{'=' * 80}")
        
        try:
            if process_single_crop(image_path, Config.OUTPUT_DIR, method=method):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("POST-PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total images: {len(image_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nCentered crops saved in: {Config.OUTPUT_DIR}")
    print("=" * 80)

# ========================================================================================================
# MAIN ENTRY POINT
# ========================================================================================================

def main():
    """Main entry point"""
    import sys
    
    # Default method
    method = 'tight'  # 'tight' or 'centered'
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help':
            print("\nUSAGE:")
            print("  python postprocess_center_crops.py              # Tight crop (default)")
            print("  python postprocess_center_crops.py --centered   # Centered with extra padding")
            print("\nCONFIGURATION:")
            print(f"  Input: {Config.INPUT_DIR}")
            print(f"  Output: {Config.OUTPUT_DIR}")
            print(f"  Padding: {Config.PADDING}px")
            print(f"  Method: Auto-size to fit clothing dimensions")
            return
        elif sys.argv[1] == '--centered':
            method = 'centered'
            print("[INFO] Using centered method with extra padding")
        else:
            print(f"[WARN] Unknown argument: {sys.argv[1]}")
            print("[INFO] Using default tight crop method")
    
    # Run batch processing
    batch_process_crops(method=method)

if __name__ == "__main__":
    main()




