"""
Alternative clustering methods for comparison with unsupervised CNN approach.

This module provides various clustering algorithms (K-means, DBSCAN, GMM, etc.)
for benchmarking against the CNN-based segmentation method.
"""

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster


def display_image(title, image_data):
    """Display an image using matplotlib."""
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.imshow(image_data)
    plt.axis('off')
    plt.show()


def preprocess_image(image):
    """
    Preprocess image for clustering.
    
    Args:
        image (numpy.ndarray): Input image in BGR format
        
    Returns:
        tuple: (standardized_pixels, original_shape)
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape to list of pixels
    image_reshaped = image_rgb.reshape((-1, 3))
    
    # Standardize features
    scaler = StandardScaler()
    image_standardized = scaler.fit_transform(image_reshaped)
    
    return image_standardized, image_rgb.shape


def kmeans_segmentation(image_standardized, shape, n_clusters=3):
    """
    Perform K-means clustering segmentation.
    
    Args:
        image_standardized (numpy.ndarray): Preprocessed image pixels
        shape (tuple): Original image shape
        n_clusters (int): Number of clusters
        
    Returns:
        tuple: (segmented_image, execution_time)
    """
    print(f'Executing K-means clustering with {n_clusters} clusters...')
    start_time = time.time()
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(image_standardized)
    segmented = labels.reshape(shape[:2])
    
    elapsed = time.time() - start_time
    print(f'K-means completed in {elapsed:.2f} seconds')
    
    return segmented, elapsed


def dbscan_segmentation(image_standardized, shape, eps=0.5, min_samples=5):
    """
    Perform DBSCAN clustering segmentation.
    
    Args:
        image_standardized (numpy.ndarray): Preprocessed image pixels
        shape (tuple): Original image shape
        eps (float): Maximum distance between samples
        min_samples (int): Minimum samples in a neighborhood
        
    Returns:
        tuple: (segmented_image, execution_time) or (None, elapsed) on failure
    """
    print(f'Executing DBSCAN clustering (eps={eps}, min_samples={min_samples})...')
    start_time = time.time()
    
    try:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(image_standardized)
        segmented = labels.reshape(shape[:2])
        
        elapsed = time.time() - start_time
        print(f'DBSCAN completed in {elapsed:.2f} seconds')
        print(f'Found {len(np.unique(labels))} clusters')
        
        return segmented, elapsed
    except MemoryError:
        elapsed = time.time() - start_time
        print(f'DBSCAN failed after {elapsed:.2f} seconds due to memory error')
        return None, elapsed


def gmm_segmentation(image_standardized, shape, n_components=5):
    """
    Perform Gaussian Mixture Model segmentation.
    
    Args:
        image_standardized (numpy.ndarray): Preprocessed image pixels
        shape (tuple): Original image shape
        n_components (int): Number of Gaussian components
        
    Returns:
        tuple: (segmented_image, execution_time)
    """
    print(f'Executing GMM clustering with {n_components} components...')
    start_time = time.time()
    
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    labels = gmm.fit_predict(image_standardized)
    segmented = labels.reshape(shape[:2])
    
    elapsed = time.time() - start_time
    print(f'GMM completed in {elapsed:.2f} seconds')
    
    return segmented, elapsed


def hierarchical_segmentation(image_standardized, shape, n_clusters=5, method='ward'):
    """
    Perform hierarchical clustering segmentation.
    
    Args:
        image_standardized (numpy.ndarray): Preprocessed image pixels
        shape (tuple): Original image shape
        n_clusters (int): Number of clusters
        method (str): Linkage method ('ward', 'complete', 'average', 'single')
        
    Returns:
        tuple: (segmented_image, execution_time)
        
    Warning:
        This method can be memory-intensive for large images.
    """
    print(f'Executing Hierarchical clustering with method={method}...')
    start_time = time.time()
    
    # Subsample for large images to avoid memory issues
    if len(image_standardized) > 10000:
        print(f'  Subsampling from {len(image_standardized)} to 10000 pixels')
        indices = np.random.choice(len(image_standardized), 10000, replace=False)
        sample = image_standardized[indices]
    else:
        sample = image_standardized
    
    linkage_matrix = linkage(sample, method=method)
    labels_sample = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')
    
    # For full image, assign each pixel to nearest cluster center
    if len(sample) < len(image_standardized):
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(sample, labels_sample)
        labels = knn.predict(image_standardized)
    else:
        labels = labels_sample
    
    segmented = labels.reshape(shape[:2])
    
    elapsed = time.time() - start_time
    print(f'Hierarchical clustering completed in {elapsed:.2f} seconds')
    
    return segmented, elapsed


def meanshift_segmentation(image_standardized, shape, quantile=0.2, n_samples=500):
    """
    Perform Mean Shift clustering segmentation.
    
    Args:
        image_standardized (numpy.ndarray): Preprocessed image pixels
        shape (tuple): Original image shape
        quantile (float): Quantile for bandwidth estimation
        n_samples (int): Number of samples for bandwidth estimation
        
    Returns:
        tuple: (segmented_image, execution_time)
    """
    print(f'Executing Mean Shift clustering...')
    start_time = time.time()
    
    print(f'  Estimating bandwidth...')
    bandwidth = estimate_bandwidth(
        image_standardized,
        quantile=quantile,
        n_samples=n_samples
    )
    print(f'  Bandwidth: {bandwidth:.4f}')
    
    mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    labels = mean_shift.fit_predict(image_standardized)
    segmented = labels.reshape(shape[:2])
    
    elapsed = time.time() - start_time
    print(f'Mean Shift completed in {elapsed:.2f} seconds')
    print(f'Found {len(np.unique(labels))} clusters')
    
    return segmented, elapsed


def compare_methods(image, methods=['kmeans', 'gmm'], display=True):
    """
    Compare multiple clustering methods on the same image.
    
    Args:
        image (numpy.ndarray): Input image
        methods (list): List of method names to compare
        display (bool): Whether to display results
        
    Returns:
        dict: Results for each method with segmentation and timing
    """
    print(f"\nComparing {len(methods)} segmentation methods...")
    print(f"Image shape: {image.shape}\n")
    
    # Preprocess once
    image_std, shape = preprocess_image(image)
    
    results = {}
    
    for method in methods:
        if method == 'kmeans':
            seg, time_taken = kmeans_segmentation(image_std, shape)
        elif method == 'dbscan':
            seg, time_taken = dbscan_segmentation(image_std, shape)
        elif method == 'gmm':
            seg, time_taken = gmm_segmentation(image_std, shape)
        elif method == 'hierarchical':
            seg, time_taken = hierarchical_segmentation(image_std, shape)
        elif method == 'meanshift':
            seg, time_taken = meanshift_segmentation(image_std, shape)
        else:
            print(f"Unknown method: {method}")
            continue
        
        results[method] = {
            'segmentation': seg,
            'time': time_taken
        }
        
        if display and seg is not None:
            display_image(f'{method.upper()} Segmentation', seg)
        
        print()
    
    return results


if __name__ == "__main__":
    """Example usage."""
    from src.preprocessing.masking import mask_and_crop
    
    # Load and preprocess image
    img_path = "../../data/images/DSC01902.JPG"
    img = mask_and_crop(img_path)
    
    print(f"Image shape: {img.shape}")
    
    # Compare different methods
    methods = ['kmeans', 'gmm']  # Add more: 'dbscan', 'hierarchical', 'meanshift'
    results = compare_methods(img, methods=methods, display=True)
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for method, result in results.items():
        if result['segmentation'] is not None:
            n_clusters = len(np.unique(result['segmentation']))
            print(f"{method.upper():15s}: {result['time']:6.2f}s, {n_clusters} clusters")
        else:
            print(f"{method.upper():15s}: Failed")

