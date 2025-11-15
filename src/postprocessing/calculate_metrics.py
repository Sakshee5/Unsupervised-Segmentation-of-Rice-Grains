"""
Script to calculate geometric and quality metrics for rice grains.

Each prediction image contains ONE rice grain with:
- Black background
- Dark grey defect pixels
- Whitish healthy grain pixels

Simple calculations based on pixel colors and positions.
"""

import os
import cv2
import numpy as np
import pandas as pd


# ============================= CONFIGURATION =============================

# Color definitions (in BGR format) - must match normalize_colors.py
DEFECT_COLOR_BGR = (99, 96, 90)          # Dark grey for defects
HEALTHY_RICE_COLOR_BGR = (245, 237, 211)  # Whitish rice color
BACKGROUND_COLOR_BGR = (0, 0, 0)         # Black background

# Output CSV filename
OUTPUT_CSV = "geometric_metrics_consolidated.csv"

# =========================================================================


def calculate_grain_metrics(image_path, image_name, defect_color, healthy_color, background_color):
    """
    Calculate metrics for a single rice grain image.
    
    Args:
        image_path (str): Path to the image file
        image_name (str): Name of the image (for CSV output)
        defect_color (tuple): BGR color for defects
        healthy_color (tuple): BGR color for healthy grain
        background_color (tuple): BGR color for background
        
    Returns:
        dict: Dictionary containing all computed metrics, or None if error
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"  ✗ Could not read: {image_name}")
        return None
    
    height, width = image.shape[:2]
    
    # Create masks for each color
    defect_mask = np.all(image == defect_color, axis=-1)
    healthy_mask = np.all(image == healthy_color, axis=-1)
    background_mask = np.all(image == background_color, axis=-1)
    
    # Grain mask = anything not background
    grain_mask = ~background_mask
    
    # Count pixels
    defect_pixels = np.sum(defect_mask)
    healthy_pixels = np.sum(healthy_mask)
    total_grain_pixels = defect_pixels + healthy_pixels
    
    # Calculate damage area percentage
    if total_grain_pixels > 0:
        damage_area_percent = (defect_pixels / total_grain_pixels) * 100
    else:
        damage_area_percent = 0.0
    
    # Get grain pixel coordinates (for length/breadth calculation)
    grain_coords = np.argwhere(grain_mask)  # Returns (row, col) = (y, x)
    
    if len(grain_coords) == 0:
        print(f"  ✗ No grain pixels found in: {image_name}")
        return None
    
    y_coords_grain = grain_coords[:, 0]
    x_coords_grain = grain_coords[:, 1]
    
    # Calculate bounding box for length and breadth (using all grain pixels)
    y_min, y_max = y_coords_grain.min(), y_coords_grain.max()
    x_min, x_max = x_coords_grain.min(), x_coords_grain.max()
    
    length = y_max - y_min + 1  # Height of bounding box
    breadth = x_max - x_min + 1  # Width of bounding box
    
    # Get DEFECT pixel coordinates for centroid and deviation calculations
    defect_coords = np.argwhere(defect_mask)  # Returns (row, col) = (y, x)
    
    if len(defect_coords) > 0:
        # Calculate centroid and deviations ONLY for defect pixels
        y_coords_defect = defect_coords[:, 0]
        x_coords_defect = defect_coords[:, 1]
        
        x_centroid = np.mean(x_coords_defect)
        y_centroid = np.mean(y_coords_defect)
        x_deviation = np.std(x_coords_defect)
        y_deviation = np.std(y_coords_defect)
    else:
        # No defects - set to 0 or center of grain
        x_centroid = 0.0
        y_centroid = 0.0
        x_deviation = 0.0
        y_deviation = 0.0
    
    # Compile metrics
    metrics = {
        'Image': image_name,
        'Length': length,
        'Breadth': breadth,
        'Total Pixels': total_grain_pixels,
        'Damage Pixels': defect_pixels,
        'Damage Area': damage_area_percent,
        'x centroid': x_centroid,
        'y centroid': y_centroid,
        'x deviation': x_deviation,
        'y deviation': y_deviation,
    }
    
    return metrics


def calculate_metrics_for_all(input_dir, output_csv, defect_color, 
                              healthy_color, background_color):
    """
    Calculate metrics for all images in a directory and save to CSV.
    
    Args:
        input_dir (str): Directory containing post-processed predictions
        output_csv (str): Path to output CSV file
        defect_color (tuple): BGR color for defects
        healthy_color (tuple): BGR color for healthy grain
        background_color (tuple): BGR color for background
    """
    # Get all image files
    image_files = [f for f in os.listdir(input_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"\n{'='*60}")
    print(f"CALCULATING METRICS FOR {len(image_files)} IMAGES")
    print(f"{'='*60}\n")
    
    all_metrics = []
    processed_count = 0
    skipped_count = 0
    
    for img_name in image_files:
        img_path = os.path.join(input_dir, img_name)
        
        # Calculate metrics for this grain
        metrics = calculate_grain_metrics(
            img_path, img_name, defect_color, healthy_color, background_color
        )
        
        if metrics is not None:
            all_metrics.append(metrics)
            processed_count += 1
            
            if processed_count % 10 == 0:
                print(f"  Processed {processed_count}/{len(image_files)} images...")
        else:
            skipped_count += 1
    
    # Convert to DataFrame
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        
        # Reorder columns for readability
        column_order = [
            'Image', 'Length', 'Breadth', 'Total Pixels',
            'Damage Pixels', 'Damage Area', 'x centroid', 'y centroid',
            'x deviation', 'y deviation'
        ]
        df = df[column_order]
        
        # Save to CSV
        df.to_csv(output_csv, index=False)
        
        print(f"\n{'='*60}")
        print(f"COMPLETED")
        print(f"{'='*60}")
        print(f"  Total images processed: {processed_count}")
        print(f"  Skipped: {skipped_count}")
        print(f"  Output saved to: {output_csv}")
        
        # Print summary statistics
        print(f"\n{'='*60}")
        print(f"SUMMARY STATISTICS")
        print(f"{'='*60}")
        print(f"  Average damage area: {df['Damage Area'].mean():.2f}%")
        print(f"  Median damage area: {df['Damage Area'].median():.2f}%")
        print(f"  Max damage area: {df['Damage Area'].max():.2f}%")
        print(f"  Min damage area: {df['Damage Area'].min():.2f}%")
        print(f"  Grains with >20% damage: {(df['Damage Area'] > 20).sum()}")
        print(f"  Grains with >40% damage: {(df['Damage Area'] > 40).sum()}\n")
    else:
        print(f"\n✗ No valid grains processed!\n")


def main():
    """Main execution function."""
    # Set up paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_dir = os.path.join(base_dir, "data", "predictions-post-processed")
    output_csv_path = os.path.join(base_dir, "data", OUTPUT_CSV)
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"\nERROR: Input directory not found: {input_dir}")
        print("Please run normalize_colors.py first to create post-processed predictions.\n")
        return
    
    # Calculate metrics
    calculate_metrics_for_all(
        input_dir=input_dir,
        output_csv=output_csv_path,
        defect_color=DEFECT_COLOR_BGR,
        healthy_color=HEALTHY_RICE_COLOR_BGR,
        background_color=BACKGROUND_COLOR_BGR
    )


if __name__ == "__main__":
    main()
