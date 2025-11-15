"""
Utility functions for color processing and image manipulation.

This module provides functions to quantize colors, find unique colors,
and replace colors in segmented images.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans


def quantize_colors(image, n_colors=3):
    """
    Quantize an image to a specified number of colors using K-means clustering.
    
    This is essential for handling minor color variations in segmented images
    and ensuring consistent color mapping across predictions.
    
    Args:
        image (numpy.ndarray): Input image in BGR format (height, width, 3)
        n_colors (int): Number of colors to reduce the image to
        
    Returns:
        tuple: (quantized_image, unique_colors)
            - quantized_image (numpy.ndarray): Image with reduced colors (uint8, BGR)
            - unique_colors (numpy.ndarray): Array of BGR values (n_colors, 3)
    """
    # Reshape the image to be a list of pixels
    pixels = image.reshape(-1, 3)
    
    # Apply KMeans to find the top n_colors in the image
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    # Replace each pixel with its closest cluster center
    # IMPORTANT: Must be uint8 for OpenCV to save/display properly
    new_colors = kmeans.cluster_centers_.astype(np.uint8)
    labels = kmeans.labels_
    quantized_image = new_colors[labels].reshape(image.shape).astype(np.uint8)
    
    return quantized_image, new_colors


def find_unique_colors(image):
    """
    Find all unique colors in an image.
    
    Useful for analyzing segmentation results and identifying distinct regions.
    
    Args:
        image (numpy.ndarray): Input image in BGR format
        
    Returns:
        list: List of tuples containing unique BGR color values
    """
    # Reshape the image to be a list of pixels
    pixels = image.reshape(-1, 3)

    # Find all unique colors
    unique_colors = np.unique(pixels, axis=0)

    # Convert unique colors to a list of tuples
    unique_colors_list = [tuple(color) for color in unique_colors]
    
    return unique_colors_list


def replace_color_exact(image, target_color, replacement_color):
    """
    Replace all instances of a target color with a replacement color.
    
    Args:
        image (numpy.ndarray): Input image in BGR format
        target_color (tuple): BGR values of color to replace (B, G, R)
        replacement_color (tuple): BGR values of replacement color (B, G, R)
        
    Returns:
        numpy.ndarray: Image with colors replaced (modifies in-place)
    """
    # Ensure colors are in correct format
    target_color = np.array(target_color, dtype=np.uint8)
    replacement_color = np.array(replacement_color, dtype=np.uint8)
    
    # Create a mask where the target color matches exactly
    mask = np.all(image == target_color, axis=-1)
    
    # Replace the target color with the replacement color
    image[mask] = replacement_color

    return image


def quantize_to_fixed_palette(image, color_palette):
    """
    Quantize an image to a fixed color palette by assigning each pixel 
    to the nearest color in the palette.
    
    This ensures ALL images use the exact same colors, preventing variations
    in K-means clustering across different images.
    
    Args:
        image (numpy.ndarray): Input image in BGR format (height, width, 3)
        color_palette (list or numpy.ndarray): Fixed palette of BGR colors to use
        
    Returns:
        numpy.ndarray: Quantized image using only colors from the palette (uint8, BGR)
    """
    # Ensure palette is numpy array
    palette = np.array(color_palette, dtype=np.float32)
    
    # Reshape image to list of pixels
    pixels = image.reshape(-1, 3).astype(np.float32)
    
    # For each pixel, find the nearest color in the palette
    # Using vectorized distance calculation for speed
    quantized_labels = np.zeros(len(pixels), dtype=np.int32)
    
    for i, pixel in enumerate(pixels):
        # Calculate Euclidean distance to each palette color
        distances = np.sqrt(np.sum((palette - pixel) ** 2, axis=1))
        # Assign to nearest color
        quantized_labels[i] = np.argmin(distances)
    
    # Map labels back to colors
    quantized_pixels = palette[quantized_labels].astype(np.uint8)
    quantized_image = quantized_pixels.reshape(image.shape)
    
    return quantized_image


def print_colors_for_inspection(unique_colors):
    """
    Print unique colors in a format easy to copy-paste for manual configuration.
    
    Args:
        unique_colors (list): List of BGR color tuples
    """
    print("\n" + "="*60)
    print("UNIQUE COLORS FOUND (BGR format):")
    print("="*60)
    for i, color in enumerate(unique_colors):
        print(f"{i+1}. {color}")
    print("="*60)
    print("\nManually inspect your predictions to determine:")
    print("  - Which colors represent DEFECTS")
    print("  - Which colors represent HEALTHY GRAIN")
    print("  - Background should be (0, 0, 0)")
    print("\nThen update the color mapping in your script.\n")

