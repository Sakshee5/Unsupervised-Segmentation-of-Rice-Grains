"""
Unsupervised Segmentation Network for Rice Grain Defect Detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationNet(nn.Module):
    """
    CNN model for unsupervised image segmentation.
    
    The network consists of multiple convolutional layers with batch normalization
    that learns to segment images without labeled data by clustering similar features.
    
    Args:
        input_dim (int): Number of input channels (typically 3 for RGB images)
        n_channel (int): Number of output channels/features (default: 50)
        n_conv (int): Number of convolutional layers (default: 2)
    """
    
    def __init__(self, input_dim, n_channel=50, n_conv=2):
        super(SegmentationNet, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=input_dim,
            out_channels=n_channel,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(n_channel)
        
        # Create middle convolutional layers dynamically
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(n_conv - 1):
            self.conv2.append(
                nn.Conv2d(
                    n_channel,
                    n_channel,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1
                )
            )
            self.bn2.append(nn.BatchNorm2d(n_channel))
        
        self.conv3 = nn.Conv2d(
            n_channel,
            n_channel,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1
        )
        self.bn3 = nn.BatchNorm2d(n_channel)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output feature maps of shape (batch_size, n_channel, height, width)
        """
        # First conv layer
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        
        # Middle conv layers
        for conv, bn in zip(self.conv2, self.bn2):
            x = conv(x)
            x = F.relu(x)
            x = bn(x)
        
        # Final conv layer
        x = self.conv3(x)
        x = self.bn3(x)
        
        return x

