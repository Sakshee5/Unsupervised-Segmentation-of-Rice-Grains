"""
Train unsupervised segmentation network for rice grain defect detection.

This script trains a CNN to segment rice grain images without labeled data,
using the approach from Kanezaki et al. (2018).
"""

import argparse
import os
import random
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from skimage import segmentation
from torch.autograd import Variable

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.segmentation_net import SegmentationNet
from preprocessing.masking import mask_and_crop


def seed_everything(seed):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Unsupervised Segmentation Training for Rice Grain Quality Inspection'
    )
    parser.add_argument(
        '--image',
        type=str,
        default='data/images/DSC01902.JPG',
        help='Path to input image'
    )
    parser.add_argument(
        '--nChannel',
        type=int,
        default=10,
        help='Number of feature channels (default: 30)'
    )
    parser.add_argument(
        '--nConv',
        type=int,
        default=2,
        help='Number of convolutional layers (default: 2)'
    )
    parser.add_argument(
        '--maxIter',
        type=int,
        default=100,
        help='Maximum number of training iterations (default: 80)'
    )
    parser.add_argument(
        '--minLabels',
        type=int,
        default=3,
        help='Minimum number of labels (default: 3)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.15,
        help='Learning rate (default: 0.02)'
    )
    parser.add_argument(
        '--visualize',
        type=int,
        default=1,
        choices=[0, 1],
        help='Enable visualization during training (default: 1)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=10,
        help='Random seed for reproducibility (default: 10)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use for training (default: auto)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='models',
        help='Directory to save trained model (default: models)'
    )
    parser.add_argument(
        '--save_loss',
        action='store_true',
        help='Save loss values to Excel file'
    )
    
    return parser.parse_args()


def unsupervised_segmentation(image, args, device):
    """
    Perform unsupervised segmentation on an image.
    
    Args:
        image (numpy.ndarray): Input image in BGR format
        args: Command line arguments
        device (torch.device): Device to run training on
        
    Returns:
        SegmentationNet: Trained model
    """
    loss_lst = []
    
    # Denoise image
    print("Preprocessing image...")
    im_denoise = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 0, 15)
    
    # Convert image to tensor
    data = torch.from_numpy(
        np.array([image.transpose((2, 0, 1)).astype('float32') / 255.])
    ).to(device)
    data = Variable(data)
    
    # Generate superpixels using Felzenszwalb algorithm
    # Lower scale = more superpixels = finer segmentation
    # Higher min_size = fewer, larger superpixels = smoother segmentation
    print("Generating superpixels...")
    labels = segmentation.felzenszwalb(
        im_denoise, 
        scale=32,      # Increased for coarser segmentation
        sigma=0.5,     # Increased for more smoothing
        min_size=100   # Increased for larger superpixels
    )
    labels = labels.reshape(image.shape[0] * image.shape[1])
    u_labels = np.sort(np.unique(labels))
    
    # Create index list for each superpixel
    l_inds = []
    for i in range(len(u_labels)):
        l_inds.append(np.where(labels == u_labels[i])[0])
    
    # Initialize model
    print(f"Initializing model with {args.nChannel} channels and {args.nConv} conv layers...")
    model = SegmentationNet(
        input_dim=data.size(1),
        n_channel=args.nChannel,
        n_conv=args.nConv
    ).to(device)
    model.train()
    
    # Loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Fixed color mapping for visualization
    color_seed = 42
    np.random.seed(color_seed)
    label_colours = np.random.randint(255, size=(args.nChannel, 3))
    
    print(f"\nStarting training for {args.maxIter} iterations...")
    print("-" * 60)
    
    # Training loop
    for batch_idx in range(args.maxIter):
        # Forward pass
        optimizer.zero_grad()
        output = model(data)[0]  # (C, H, W)
        output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel)  # (H*W, C)
        
        # Argmax classification to get target labels
        _, target = torch.max(output, 1)
        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))
        
        # Refine labels using superpixel information
        for i in range(len(l_inds)):
            labels_per_sp = im_target[l_inds[i]]
            u_labels_per_sp = np.unique(labels_per_sp)
            hist = np.zeros(len(u_labels_per_sp))
            for j in range(len(hist)):
                hist[j] = len(np.where(labels_per_sp == u_labels_per_sp[j])[0])
            im_target[l_inds[i]] = u_labels_per_sp[np.argmax(hist)]
        
        target = torch.from_numpy(im_target).to(device)
        
        # Visualization
        if args.visualize:
            im_target_rgb = np.array([
                label_colours[c % args.nChannel] for c in im_target
            ])
            im_target_rgb = im_target_rgb.reshape(image.shape).astype(np.uint8)
            
            # Resize for display
            scale_percent = 50
            width = int(im_target_rgb.shape[1] * scale_percent / 100)
            height = int(im_target_rgb.shape[0] * scale_percent / 100)
            dim = (width, height)
            
            cv2.imshow('Segmentation Progress', cv2.resize(
                im_target_rgb, dim, interpolation=cv2.INTER_AREA
            ))
            cv2.waitKey(10)
        
        # Compute loss and backpropagate
        loss = loss_fn(output, target)
        
        # Add penalty for too many labels (encourages convergence)
        if nLabels > args.minLabels:
            label_penalty = 0.01 * (nLabels - args.minLabels)
            loss = loss + label_penalty
        
        loss.backward()
        optimizer.step()
        
        # Log progress
        penalty_str = f" (+penalty)" if nLabels > args.minLabels else ""
        print(f"Iter {batch_idx+1:3d}/{args.maxIter} | "
              f"Labels: {nLabels:2d} | "
              f"Loss: {loss.item():.6f}{penalty_str}")
        loss_lst.append(loss.item())
        
        # Early stopping if we reach desired number of labels and loss is stable
        if nLabels <= args.minLabels and batch_idx > 50:
            last_10_losses = loss_lst[-10:]
            if len(last_10_losses) == 10:
                loss_variance = np.var(last_10_losses)
                if loss_variance < 0.001:  # Loss is stable
                    print(f"\nEarly stopping: Reached {nLabels} labels with stable loss")
                    break
    
    print("-" * 60)
    print("Training completed!")
    
    # Save loss values if requested
    if args.save_loss:
        loss_file = f"seed_{args.seed}_loss_values.xlsx"
        df = pd.DataFrame({f"Loss (Seed {args.seed})": loss_lst})
        df.to_excel(loss_file, index=False)
        print(f"Loss values saved to {loss_file}")
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Show final result
    if args.visualize:
        cv2.imshow('Final Segmentation', cv2.resize(
            im_target_rgb, dim, interpolation=cv2.INTER_AREA
        ))
        print("\nPress any key to close visualization...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return model


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    seed_everything(args.seed)
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"\n{'='*60}")
    print("Rice Grain Unsupervised Segmentation Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Image: {args.image}")
    print(f"Seed: {args.seed}")
    print(f"{'='*60}\n")
    
    # Load and preprocess image
    print(f"Loading image from {args.image}...")
    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        return
    
    img = mask_and_crop(args.image)
    print(f"Image shape: {img.shape}")
    
    # Train model
    model = unsupervised_segmentation(img, args, device)
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()

