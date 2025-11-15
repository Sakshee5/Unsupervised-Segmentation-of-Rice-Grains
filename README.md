# Rice Grain Quality Inspection

Unsupervised defect segmentation of high-resolution (4000×6000 pixels) individual rice grain images using deep learning.

## Overview

This project implements an unsupervised CNN-based approach for automatically segmenting defects in rice grain images without requiring labeled training data. The method combines:

- **Deep Learning**: CNN feature extraction for rich image representations
- **Unsupervised Learning**: No labeled data required for training
- **Superpixel Refinement**: SLIC algorithm for improved segmentation quality
- **HSV Color Space**: Robust to lighting variations

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Configuration](#-configuration)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)

## Features

- **Unsupervised Segmentation**: No manual labeling required
- **GPU Acceleration**: CUDA support for faster training
- **Interactive HSV Tuning**: GUI tool for optimizing color masking
- **Batch Processing**: Process multiple images efficiently
- **Quality Metrics**: Built-in defect analysis and visualization
- **Multiple Clustering Methods**: Compare with K-means, GMM, DBSCAN, etc.
- **Modern Package Management**: Support for `uv` package manager

## Installation

### Option 1: Using `uv` (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

### Option 2: Using pip

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install from pyproject.toml
pip install -e .
```

## Quick Start

### 1. Prepare Your Data

Place your rice grain images in the `data/images/` directory:

```bash
data/images/
├── DSC01902.JPG
├── DSC01912.JPG
└── ...
```

### 2. (Optional) Tune HSV Values

If your images have different lighting conditions, tune the HSV masking values:

```bash
python -m src.preprocessing.hsv_tuner
```

Adjust the trackbars until the grain is properly isolated, then update the values in `src/preprocessing/masking.py`.

### 3. Train the Model

```bash
python -m src.training.train --image data/images/DSC01902.JPG --maxIter 80 --visualize 1
```

This will:
- Preprocess the image (masking and cropping)
- Train the unsupervised segmentation model
- Save the trained model to `models/model.pth`
- Display real-time visualization (if enabled)

### 4. Run Inference

Segment all images in a directory:

```bash
python -m src.inference.predict --input data/images --output data/predictions
```

Or segment a single image:

```bash
python -m src.inference.predict --input data/images/DSC01902.JPG --output data/predictions --visualize
```

## Usage

### Training

```bash
python -m src.training.train [OPTIONS]
```

**Options:**
- `--image PATH`: Path to training image (default: `data/images/DSC01902.JPG`)
- `--nChannel INT`: Number of feature channels (default: 10)
- `--nConv INT`: Number of convolutional layers (default: 2)
- `--maxIter INT`: Maximum training iterations (default: 100)
- `--lr FLOAT`: Learning rate (default: 0.15)
- `--seed INT`: Random seed for reproducibility (default: 10)
- `--visualize {0,1}`: Enable visualization (default: 1)
- `--device {auto,cpu,cuda}`: Device to use (default: auto)
- `--output_dir PATH`: Output directory for model (default: `models`)
- `--save_loss`: Save loss values to Excel file

**Example:**
```bash
# Train with custom parameters
python -m src.training.train \
    --image data/images/DSC01912.JPG \
    --maxIter 100 \
    --lr 0.01 \
    --nChannel 20 \
    --save_loss
```

### Inference

```bash
python -m src.inference.predict [OPTIONS]
```

**Options:**
- `--model PATH`: Path to trained model (default: `models/model.pth`)
- `--input PATH`: Input image or directory (default: `data/images`)
- `--output PATH`: Output directory (default: `data/predictions`)
- `--nChannel INT`: Number of channels (must match training) (default: 10)
- `--nConv INT`: Number of conv layers (must match training) (default: 2)
- `--device {auto,cpu,cuda}`: Device to use (default: auto)
- `--visualize`: Display predictions interactively
- `--seed INT`: Random seed for color mapping (default: 42)

**Example:**
```bash
# Batch processing with visualization
python -m src.inference.predict \
    --input data/images \
    --output data/predictions \
    --model models/model.pth \
    --visualize
```

### HSV Tuning

```bash
python -m src.preprocessing.hsv_tuner
```

Interactive tool with trackbars to find optimal HSV color ranges for grain masking. Press 'q' to quit and the tool will print the optimal values.

### Alternative Clustering Methods

Compare the CNN approach with traditional clustering methods:

```python
from src.utils.clustering import compare_methods
from src.preprocessing.masking import mask_and_crop

img = mask_and_crop("data/images/DSC01902.JPG")
results = compare_methods(img, methods=['kmeans', 'gmm', 'dbscan'])
```

## Project Structure

```
/
├── src/                          # Source code
│   ├── models/                   # Neural network models
│   │   ├── __init__.py
│   │   └── segmentation_net.py   # Unsupervised segmentation CNN
│   ├── preprocessing/            # Image preprocessing
│   │   ├── __init__.py
│   │   ├── masking.py            # HSV masking and cropping
│   │   └── hsv_tuner.py          # Interactive HSV tuning tool
│   ├── training/                 # Training scripts
│   │   ├── __init__.py
│   │   └── train.py              # Training pipeline
│   ├── inference/                # Inference scripts
│   │   ├── __init__.py
│   │   └── predict.py            # Prediction pipeline
│   ├── postprocessing/           # Post-processing utilities
│   │   ├── __init__.py
│   │   ├── utils.py                  # Color processing utilities
│   │   ├── normalize_colors.py       # Step 1: Color normalization script
│   │   ├── calculate_metrics.py      # Step 2: Metrics calculation script
│   │   ├── visualization.py          # Step 3: Plotting and visualization
│   │   ├── clustering.py             # Alternative clustering methods for comparison
│   │   └── README.md          
├── data/                                # Data directory
│   ├── images/                          # Input images
│   └── predictions/                     # Segmentation results
│   └── predictions-post-processing/     # Post Processed Segmentation results
├── models/                       # Trained models
├── pyproject.toml                # Project metadata and dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## Methodology

### 1. Image Preprocessing

**RGB → HSV → Grain Masking → Background Cropping**

- Convert image to HSV color space (more robust to lighting changes)
- Apply manual color detection to mask the grain
- Crop background to reduce computation
- Add padding around the grain

### 2. CNN Forward Pass

**Feature Extraction**

- Input: `(W, H, 3)` RGB image
- Output: `(W, H, C)` feature representation (C=50 by default)
- Network: Multiple convolutional layers with batch normalization

### 3. Argmax Classification

**Target Image Generation**

- Flatten feature vector: `(W×H, C)`
- Apply argmax to get cluster assignments: `(W×H, 1)`
- Reshape to image: `(W, H, 1)`

### 4. Superpixel Refinement

**Clustering with Felzenszwalb Algorithm**

- Generate superpixels using graph-based segmentation
- Refine CNN output by enforcing spatial consistency
- Assign majority label within each superpixel

### 5. Loss Computation

**Backpropagation**

- Calculate cross-entropy loss between output `(W, H, C)` and target `(W, H, 1)`
- Minimize loss through backpropagation
- Update network parameters iteratively

## Configuration

### HSV Color Ranges

Default values in `src/preprocessing/masking.py`:

```python
lower = np.array([70, 110, 0])    # H, S, V lower bounds
upper = np.array([109, 255, 255])  # H, S, V upper bounds
```

Adjust these using the HSV tuner tool for different lighting conditions.

### Model Hyperparameters

Key parameters for training (in `src/training/train.py`):

- `nChannel`: Number of feature channels (default: 50)
- `nConv`: Number of convolutional layers (default: 2)
- `maxIter`: Training iterations (default: 80)
- `lr`: Learning rate (default: 0.02)
- `minLabels`: Minimum number of segments (default: 3)

### Superpixel Parameters

Felzenszwalb algorithm settings:

```python
segmentation.felzenszwalb(image, scale=1, sigma=0.1, min_size=60)
```

### Post-Processing
Check `src/postprocessing/README.md`
```