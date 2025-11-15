"""
Image preprocessing functions for rice grain segmentation.

This module provides functions to mask and crop rice grain images using HSV color space,
which is more robust to lighting variations than RGB.
"""

import cv2
import numpy as np


def mask_and_crop(img_path, resize_dims=(800, 1200)):
    """
    Mask the rice grain and crop the background from an image.
    
    This function:
    1. Reads and resizes the image
    2. Converts to HSV color space
    3. Creates a mask to isolate the rice grain from background
    4. Removes background using morphological operations
    5. Crops to the grain region and adds padding
    
    Args:
        img_path (str): Path to the input image
        resize_dims (tuple): Target dimensions (width, height) for resizing. Default: (800, 1200)
        
    Returns:
        numpy.ndarray: Processed image with grain isolated and cropped
    """
    # Read and resize the image
    img = cv2.imread(img_path)
    img = cv2.resize(img, resize_dims)
    
    # Convert the image to HSV color space
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the HSV color range for masking the background
    # These values isolate the grain from the background
    lower = np.array([70, 110, 0])
    upper = np.array([109, 255, 255])

    # Create the mask and apply it to the image
    mask = cv2.inRange(imgHSV, lower, upper)
    imgResult = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))

    # Use morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    imgResult = cv2.morphologyEx(imgResult, cv2.MORPH_CLOSE, kernel)
    imgResult = cv2.morphologyEx(imgResult, cv2.MORPH_OPEN, kernel)

    # Trim the black borders
    def crop(frame):
        """Crop image to non-zero region."""
        coords = np.argwhere(frame[:, :, 0] != 0)
        if coords.size == 0:
            return frame
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        trimmed = frame[y0:y1, x0:x1]
        return trimmed

    final = crop(imgResult)
    
    # Add padding around the cropped grain
    final = cv2.copyMakeBorder(
        final, 10, 10, 10, 10,
        cv2.BORDER_CONSTANT,
        value=0
    )

    return final


def mask_and_crop_high_complexity(img_path, resize_dims=(800, 1200)):
    """
    Advanced masking with neighbor-based background removal.
    
    This is a more aggressive version that checks neighboring pixels
    to ensure complete background removal.
    
    Args:
        img_path (str): Path to the input image
        resize_dims (tuple): Target dimensions (width, height) for resizing
        
    Returns:
        numpy.ndarray: Processed image with grain isolated and cropped
    """
    img = cv2.resize(cv2.imread(img_path), resize_dims)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define HSV range for background masking
    lower = np.array([70, 110, 0])
    upper = np.array([109, 255, 255])

    mask = cv2.inRange(imgHSV, lower, upper)
    imgResult = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))

    def neighbours(im):
        """
        Remove pixels whose neighbors are all black (background).
        
        This function checks each pixel's surrounding neighbors up to 5 pixels away.
        If all neighbors in all 4 directions are black, the pixel is set to black.
        """
        # Set border pixels to 0
        for i in range(im.shape[1]):
            im[0, i, :] = 0
            im[im.shape[0] - 1, i, :] = 0

        for i in range(im.shape[0]):
            im[i, 0, :] = 0
            im[i, im.shape[1] - 1, :] = 0

        # Check neighbors up to 5 pixels away
        neighbours_range = [1, 2, 3, 4, 5]
        for val in neighbours_range:
            for i in range(val, im.shape[0] - val):
                for j in range(val, im.shape[1] - val):
                    if np.any(im[i, j, :]) != 0:
                        if (np.all(im[i + val, j, :] == 0) and
                            np.all(im[i, j + val, :] == 0) and
                            np.all(im[i - val, j, :] == 0) and
                            np.all(im[i, j - val, :] == 0)):
                            im[i, j, :] = 0

        return im

    def trim(frame):
        """Recursively trim black borders."""
        if not np.sum(frame[0]):
            return trim(frame[1:])
        elif not np.sum(frame[-1]):
            return trim(frame[:-2])
        elif not np.sum(frame[:, 0]):
            return trim(frame[:, 1:])
        elif not np.sum(frame[:, -1]):
            return trim(frame[:, :-2])
        return frame

    final = cv2.copyMakeBorder(
        trim(neighbours(imgResult)),
        10, 10, 10, 10,
        cv2.BORDER_CONSTANT,
        None,
        value=0
    )

    return final


if __name__ == "__main__":
    """Example usage."""
    import time
    
    img_path = r"../../data/images/DSC01912.JPG"
    
    print("Processing image...")
    start = time.time()
    img = mask_and_crop(img_path)
    end = time.time()
    
    print(f"Processing completed in {end - start:.4f} seconds")
    print(f"Output shape: {img.shape}")
    
    cv2.imshow('Processed Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

