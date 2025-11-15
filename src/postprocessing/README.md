# Post-Processing Pipeline

This module provides a complete pipeline for post-processing unsupervised segmentation results and calculating quality metrics for rice grain analysis.

## Overview

The pipeline consists of three main steps:

1. **Color Normalization** - Standardize colors across all predictions
2. **Metrics Calculation** - Compute geometric and defect metrics per grain
3. **Visualization** - Generate plots and analysis

## Directory Structure

```
src/postprocessing/
├── utils.py                  # Color processing utilities
├── normalize_colors.py       # Step 1: Color normalization script
├── calculate_metrics.py      # Step 2: Metrics calculation script
├── visualization.py          # Step 3: Plotting and visualization
├── clustering.py            # Alternative clustering methods for comparison
└── README.md                # This file
```

## Workflow

### Step 1: Inspect Colors

When you have a trained model and its predictions in `data/predictions/`, you need to identify which colors represent defects vs healthy grain.

```bash
python src/postprocessing/normalize_colors.py
```

**Initial Configuration:**
- Set `INSPECT_MODE = True` (default)
- Set `SAMPLE_IMAGE` to any prediction image filename
- Run the script

**Output:**
- Prints a fixed `COLOR_PALETTE` to copy (ensures ALL images use identical colors)
- Saves a quantized sample image for visual inspection
- Shows you which colors to map

### Step 2: Configure Color Mapping

Open `normalize_colors.py` and update with the printed palette:

```python
# Copy the exact palette from inspection output
COLOR_PALETTE = [
    (0, 0, 0),        # Background
    (14, 106, 70),    # Color 1
    (188, 19, 101),   # Color 2
    (235, 156, 36),   # Color 3
]

# Then specify which are defects vs healthy
DEFECT_COLORS_TO_REPLACE = [
    (14, 106, 70), 
    (235, 156, 36)
]

HEALTHY_COLORS_TO_REPLACE = [
    (188, 19, 101),
]
```

**Important:** `COLOR_PALETTE` ensures all images use the exact same colors (no variations like `(14,106,70)` vs `(15,106,71)`)

### Step 3: Process All Predictions

```python
# In normalize_colors.py
INSPECT_MODE = False
```

Run again:
```bash
python src/postprocessing/normalize_colors.py
```

**Output:**
- Processes all images in `data/predictions/`
- Normalizes colors to:
  - Defects → Dark grey `(97, 59, 48)`
  - Healthy grain → Whitish rice `(245, 237, 211)`
  - Background → Black `(0, 0, 0)`
- Saves to `data/predictions-post-processed/`

### Step 4: Calculate Metrics

```bash
python src/postprocessing/calculate_metrics.py
```

**What it does:**
- Segments individual grains using connected components
- For each grain, calculates:
  - **Geometric:** Length, Breadth, Total Pixels
  - **Defect Analysis:** Damage Pixels, Damage Area %
  - **Position:** x/y centroids
  - **Shape:** x/y deviations, normalized y deviation
- Saves to `data/geometric_metrics_consolidated.csv`

**Configuration:**
```python
MIN_GRAIN_SIZE = 500  # Minimum pixels to consider a grain
```

### Step 5: Visualize Results (Optional)

```python
from src.postprocessing.visualization import plot_damage_area_histogram

plot_damage_area_histogram(
    "data/geometric_metrics_consolidated.csv",
    output_path="damage_distribution.png"
)
```

## Important Notes

### Color Format: BGR vs RGB
- OpenCV uses **BGR format**: `(Blue, Green, Red)`
- When you see color `(14, 106, 70)`, it means:
  - Blue: 14
  - Green: 106
  - Red: 70

### Fixed Color Palette
All images are quantized to a **fixed palette** (not per-image K-means):
- Inspection mode determines the palette from a sample image
- Processing mode maps every pixel to the nearest palette color
- Guarantees identical colors across all predictions (e.g., all defects become exactly `(14, 106, 70)`)
- Prevents issues where some images have `(14, 106, 70)` and others have `(15, 106, 71)`

### Model-Specific Colors
- **Each trained model converges to different colors**
- Colors are consistent within a model's predictions
- You must reconfigure for each new trained model

### Handling Multiple Models
If you train multiple models:
1. Create separate directories: `data/predictions_model1/`, `data/predictions_model2/`
2. Configure color mappings for each model
3. Process separately or create config files for each model

## Example: Complete Workflow

```bash
# 1. Train model and run inference (done separately)
# This creates: data/predictions/*.png

# 2. Inspect colors
cd src/postprocessing
python normalize_colors.py  # INSPECT_MODE=True

# 3. Update color mappings in normalize_colors.py
# Set DEFECT_COLORS_TO_REPLACE and HEALTHY_COLORS_TO_REPLACE

# 4. Normalize all predictions
python normalize_colors.py  # INSPECT_MODE=False

# 5. Calculate metrics
python calculate_metrics.py

# 6. Check results
# - Post-processed images: data/predictions-post-processed/
# - Metrics CSV: data/geometric_metrics_consolidated.csv
```

## CSV Output Format

The `geometric_metrics_consolidated.csv` contains:

| Column | Description |
|--------|-------------|
| Image | Source image filename |
| Length | Height of grain bounding box (pixels) |
| Breadth | Width of grain bounding box (pixels) |
| Total Pixels | Total grain area |
| Damage Pixels | Number of defect pixels |
| Damage Area | Percentage of damage (0-100) |
| x centroid | Center x coordinate of defect pixels |
| y centroid | Center y coordinate of defect pixels |
| x deviation | Std dev of defect x positions (spread along width) |
| y deviation | Std dev of defect y positions (spread along height) |

**Note:** Centroid and deviation metrics are calculated for **defect pixels only** - they show where damage is located and how spread out it is on the grain.

## Troubleshooting

**No grains found:**
- Verify post-processed images have the correct colors
- Ensure background is pure black `(0, 0, 0)`
- Check that defect/healthy colors match your configuration

**Colors not matching:**
- Re-run inspection mode to verify colors
- Make sure you're using BGR format (not RGB)
- Check that quantization produces expected colors

**Memory issues:**
- Process images in smaller batches
- Reduce image resolution if needed

## Functions Reference

### utils.py
- `quantize_colors(image, n_colors)` - Reduce image to N colors using K-means (used in inspection)
- `quantize_to_fixed_palette(image, palette)` - Map image to fixed color palette (used in processing)
- `find_unique_colors(image)` - Get all unique colors in image
- `replace_color_exact(image, target, replacement)` - Replace colors exactly
- `print_colors_for_inspection(colors)` - Pretty-print colors for manual review

### normalize_colors.py
- `inspect_sample_image()` - Analyze one image for color identification
- `normalize_predictions()` - Process all predictions with color mapping

### calculate_metrics.py
- `calculate_grain_metrics()` - Compute all metrics for one grain image
- `calculate_metrics_for_all()` - Batch process entire directory

**Note:** Each prediction image contains exactly one rice grain

### visualization.py
- `plot_damage_area_histogram()` - Visualize damage distribution
- `plot_metric_histogram()` - Generic metric visualization
- `plot_loss_curves()` - Training loss visualization
- `visualize_segmentation_comparison()` - Side-by-side comparison

