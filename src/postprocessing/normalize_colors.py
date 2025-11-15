"""
Script to normalize colors in segmented predictions.

This script takes raw predictions from unsupervised segmentation and normalizes
the colors to a consistent scheme:
- Defect pixels -> dark grey
- Healthy grain pixels -> whitish rice color
- Background (black) -> stays black

Usage:
    1. Run this script with INSPECT_MODE = True to see unique colors
    2. Manually set the color mappings below
    3. Run again with INSPECT_MODE = False to process all images
"""

import os
import cv2
import numpy as np
from utils import (
    quantize_colors, 
    quantize_to_fixed_palette,
    find_unique_colors, 
    replace_color_exact, 
    print_colors_for_inspection
)


# ============================= CONFIGURATION =============================

# Set to True to inspect colors, False to process all images
INSPECT_MODE = False

# Sample image to inspect (used when INSPECT_MODE = True)
SAMPLE_IMAGE = "DSC01902.JPG.png"

# Number of colors to quantize to (typically 3: background, healthy, defect)
N_COLORS = 4

# Define standard output colors (in BGR format)
DEFECT_COLOR_BGR = (99, 96, 90)          # Dark grey for defects
HEALTHY_RICE_COLOR_BGR = (245, 237, 211)  # Whitish rice color for healthy grain
BACKGROUND_COLOR_BGR = (0, 0, 0)         # Black background stays black

# ============================= COLOR MAPPING =============================
# MANUALLY SET THESE AFTER INSPECTING COLORS

# IMPORTANT: This palette MUST match exactly what you see in inspection mode
# These are the ONLY colors that will exist after quantization
COLOR_PALETTE = [
    (0, 0, 0),        # Background - always black
    (14, 106, 70),    # Color 1 - UPDATE after inspection
    (188, 19, 101),   # Color 2 - UPDATE after inspection  
    (235, 156, 36),   # Color 3 - UPDATE after inspection
]

# Now specify which colors from the palette are defects vs healthy
# These MUST be colors that exist in COLOR_PALETTE above
DEFECT_COLORS_TO_REPLACE = [
    (14, 106, 70), 
    (235, 156, 36)
]

HEALTHY_COLORS_TO_REPLACE = [
    (188, 19, 101),
]

# =========================================================================


def inspect_sample_image(predictions_dir, sample_filename, n_colors):
    """
    Inspect a sample image to identify unique colors for manual mapping.
    
    This creates a FIXED COLOR PALETTE that will be used for ALL images.
    
    Args:
        predictions_dir (str): Directory containing predictions
        sample_filename (str): Name of sample file to inspect
        n_colors (int): Number of colors to quantize to
    """
    sample_path = os.path.join(predictions_dir, sample_filename)
    
    if not os.path.exists(sample_path):
        print(f"ERROR: Sample image not found: {sample_path}")
        print(f"Available images in {predictions_dir}:")
        for f in os.listdir(predictions_dir)[:10]:
            print(f"  - {f}")
        return None
    
    print(f"\n{'='*60}")
    print(f"INSPECTING: {sample_filename}")
    print(f"{'='*60}\n")
    
    # Load and quantize
    img = cv2.imread(sample_path)
    print(f"Original image shape: {img.shape}")
    
    quantized_img, cluster_colors = quantize_colors(img, n_colors=n_colors)
    
    # Find unique colors in quantized image
    unique_colors = find_unique_colors(quantized_img)
    
    print(f"\nQuantized to {n_colors} colors.")
    print(f"\n{'='*60}")
    print("FIXED COLOR PALETTE (BGR format):")
    print("="*60)
    print("\nCopy this EXACT list to COLOR_PALETTE in the script:")
    print("\nCOLOR_PALETTE = [")
    for color in unique_colors:
        print(f"    {color},")
    print("]")
    print("\n" + "="*60)
    
    print_colors_for_inspection(unique_colors)
    
    # Save quantized sample for visual inspection
    output_dir = os.path.dirname(predictions_dir)
    output_path = os.path.join(output_dir, f"_sample_quantized_{sample_filename}")
    cv2.imwrite(output_path, quantized_img)
    print(f"\nQuantized sample saved to: {output_path}")
    print("Open this image to visually identify which colors are defects vs healthy grain.\n")
    
    return unique_colors


def normalize_predictions(predictions_dir, output_dir, color_palette, 
                         defect_colors, healthy_colors,
                         defect_output_color, healthy_output_color):
    """
    Normalize all predictions by replacing colors according to the mapping.
    
    Uses a FIXED COLOR PALETTE to ensure all images have identical colors.
    
    Args:
        predictions_dir (str): Directory containing raw predictions
        output_dir (str): Directory to save normalized predictions
        color_palette (list): Fixed BGR color palette to use for ALL images
        defect_colors (list): List of BGR tuples representing defect colors
        healthy_colors (list): List of BGR tuples representing healthy colors
        defect_output_color (tuple): BGR color to use for defects
        healthy_output_color (tuple): BGR color to use for healthy grain
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Validate that defect/healthy colors are in the palette
    palette_set = set(tuple(c) for c in color_palette)
    for color in defect_colors:
        if tuple(color) not in palette_set:
            print(f"WARNING: Defect color {color} not in COLOR_PALETTE!")
    for color in healthy_colors:
        if tuple(color) not in palette_set:
            print(f"WARNING: Healthy color {color} not in COLOR_PALETTE!")
    
    # Get all image files
    image_files = [f for f in os.listdir(predictions_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"\n{'='*60}")
    print(f"NORMALIZING {len(image_files)} PREDICTIONS")
    print(f"{'='*60}\n")
    
    print(f"Configuration:")
    print(f"  Fixed color palette: {color_palette}")
    print(f"  Defect colors to replace: {defect_colors}")
    print(f"  Healthy colors to replace: {healthy_colors}")
    print(f"  Output defect color: {defect_output_color}")
    print(f"  Output healthy color: {healthy_output_color}\n")
    
    processed_count = 0
    skipped_count = 0
    
    for img_name in image_files:
        img_path = os.path.join(predictions_dir, img_name)
        
        try:
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                print(f"  ✗ {img_name}: Could not read")
                skipped_count += 1
                continue
            
            # Quantize to FIXED palette - ensures all images have exact same colors
            quantized_img = quantize_to_fixed_palette(img, color_palette)
            
            # Replace defect colors
            for color in defect_colors:
                quantized_img = replace_color_exact(quantized_img, tuple(color), defect_output_color)
            
            # Replace healthy grain colors
            for color in healthy_colors:
                quantized_img = replace_color_exact(quantized_img, tuple(color), healthy_output_color)
            
            # Save normalized image
            output_path = os.path.join(output_dir, img_name)
            cv2.imwrite(output_path, quantized_img)
            
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"  Processed {processed_count}/{len(image_files)}...")
                
        except Exception as e:
            print(f"  ✗ {img_name}: Error - {e}")
            skipped_count += 1
    
    print(f"\n{'='*60}")
    print(f"COMPLETED")
    print(f"{'='*60}")
    print(f"  ✓ Successfully processed: {processed_count}")
    print(f"  ✗ Skipped: {skipped_count}")
    print(f"  Output directory: {output_dir}\n")


def main():
    """Main execution function."""
    # Set up paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    predictions_dir = os.path.join(base_dir, "data", "predictions")
    output_dir = os.path.join(base_dir, "data", "predictions-post-processed")
    
    if INSPECT_MODE:
        # Inspection mode: analyze sample image to identify colors
        palette = inspect_sample_image(predictions_dir, SAMPLE_IMAGE, N_COLORS)
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("1. Open the quantized sample image")
        print("2. Copy the COLOR_PALETTE list shown above")
        print("3. Update COLOR_PALETTE in the script configuration")
        print("4. Update DEFECT_COLORS_TO_REPLACE and HEALTHY_COLORS_TO_REPLACE")
        print("   (these must be colors from the COLOR_PALETTE)")
        print("5. Set INSPECT_MODE = False")
        print("6. Run this script again to process all images")
        print("\nIMPORTANT: The COLOR_PALETTE ensures ALL images use the EXACT same colors!\n")
    else:
        # Processing mode: normalize all predictions
        if not DEFECT_COLORS_TO_REPLACE and not HEALTHY_COLORS_TO_REPLACE:
            print("\nERROR: Color mappings not configured!")
            print("Please set DEFECT_COLORS_TO_REPLACE and HEALTHY_COLORS_TO_REPLACE")
            print("Run with INSPECT_MODE = True first to identify colors.\n")
            return
        
        normalize_predictions(
            predictions_dir=predictions_dir,
            output_dir=output_dir,
            color_palette=COLOR_PALETTE,
            defect_colors=DEFECT_COLORS_TO_REPLACE,
            healthy_colors=HEALTHY_COLORS_TO_REPLACE,
            defect_output_color=DEFECT_COLOR_BGR,
            healthy_output_color=HEALTHY_RICE_COLOR_BGR
        )


if __name__ == "__main__":
    main()

