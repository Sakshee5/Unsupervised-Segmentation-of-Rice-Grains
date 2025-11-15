"""
Inference script for testing trained unsupervised segmentation model.

This script loads a trained model and performs segmentation on test images.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import tqdm
from torch.autograd import Variable

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.segmentation_net import SegmentationNet
from preprocessing.masking import mask_and_crop


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Inference for Rice Grain Unsupervised Segmentation'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/model.pth',
        help='Path to trained model weights'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/images',
        help='Path to input image or directory of images'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/predictions',
        help='Directory to save predictions'
    )
    parser.add_argument(
        '--nChannel',
        type=int,
        default=10,
        help='Number of feature channels (must match training)'
    )
    parser.add_argument(
        '--nConv',
        type=int,
        default=2,
        help='Number of convolutional layers (must match training)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use for inference (default: auto)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Display predictions interactively'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for color mapping (default: 42)'
    )
    
    return parser.parse_args()


def predict_single_image(image_path, model, args, label_colours, device):
    """
    Perform segmentation on a single image.
    
    Args:
        image_path (str): Path to input image
        model (SegmentationNet): Trained model
        args: Command line arguments
        label_colours (numpy.ndarray): Color palette for visualization
        device (torch.device): Device to run inference on
        
    Returns:
        numpy.ndarray: Segmented image with color-coded regions
    """
    # Load and preprocess image
    im = mask_and_crop(image_path)
    
    # Store background pixel coordinates
    indices = np.where(im == 0)
    coordinates = list(zip(indices[0], indices[1]))
    
    # Convert to tensor
    data = torch.from_numpy(
        np.array([im.transpose((2, 0, 1)).astype('float32') / 255.])
    )
    data = Variable(data).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(data)[0]
        output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel)
        _, target = torch.max(output, 1)
    
    # Convert to color image
    im_target = target.data.cpu().numpy()
    im_target_rgb = np.array([
        label_colours[c % args.nChannel] for c in im_target
    ])
    im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)
    
    # Restore background to black
    for coord in coordinates:
        im_target_rgb[coord[0], coord[1], :] = 0
    
    return im_target_rgb


def main():
    """Main inference function."""
    args = parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"\n{'='*60}")
    print("Rice Grain Unsupervised Segmentation Inference")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        print("Please train a model first using train.py")
        return
    
    # Initialize model
    print("Loading model...")
    model = SegmentationNet(
        input_dim=3,
        n_channel=args.nChannel,
        n_conv=args.nConv
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print("Model loaded successfully!")
    
    # Set up color mapping
    np.random.seed(args.seed)
    label_colours = np.random.randint(255, size=(args.nChannel, 3))
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Get list of images to process
    if os.path.isfile(args.input):
        # Single image
        image_list = [args.input]
    elif os.path.isdir(args.input):
        # Directory of images
        image_list = [
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]
    else:
        print(f"Error: Input path not found: {args.input}")
        return
    
    if not image_list:
        print(f"No images found in {args.input}")
        return
    
    print(f"\nProcessing {len(image_list)} image(s)...\n")
    
    # Process each image
    for img_path in tqdm.tqdm(image_list, desc="Segmenting images"):
        try:
            # Perform segmentation
            im_segmented = predict_single_image(
                img_path, model, args, label_colours, device
            )
            
            # Save result
            output_filename = os.path.basename(img_path) + '.png'
            output_path = os.path.join(args.output, output_filename)
            cv2.imwrite(output_path, im_segmented)
            
            # Visualize if requested
            if args.visualize:
                cv2.imshow('Segmentation Result', im_segmented)
                key = cv2.waitKey(0)
                if key == ord('q'):
                    break
                    
        except Exception as e:
            print(f"\nError processing {img_path}: {str(e)}")
            continue
    
    if args.visualize:
        cv2.destroyAllWindows()
    
    print(f"\n{'='*60}")
    print(f"Segmentation complete!")
    print(f"Results saved to: {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

