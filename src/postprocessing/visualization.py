"""
Visualization utilities for analyzing segmentation results.

This module provides functions for plotting histograms and visualizing
quality metrics from segmentation results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_damage_area_histogram(excel_path, output_path=None):
    """
    Plot histogram of damage area distribution.
    
    Args:
        excel_path (str): Path to Excel file with 'Damage Area' column
        output_path (str, optional): Path to save figure. If None, displays plot.
    """
    df = pd.read_excel(excel_path)
    
    if 'Damage Area' not in df.columns:
        raise ValueError("Excel file must contain 'Damage Area' column")
    
    # Create area range categories
    df['Area_Range'] = pd.cut(
        df['Damage Area'],
        bins=[0, 20, 40, np.inf],
        labels=['<20%', '20-40%', '>40%']
    )
    
    # Calculate counts
    counts = df['Area_Range'].value_counts()
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(
        df,
        x='Damage Area',
        hue='Area_Range',
        bins=50,
        palette='viridis',
        multiple="stack"
    )
    
    # Add threshold lines
    plt.axvline(20, color='r', linestyle='--', linewidth=2)
    plt.axvline(40, color='b', linestyle='--', linewidth=2)
    
    # Add labels
    plt.text(20.5, plt.ylim()[1] * 0.9, '20%', rotation=0, color='r', fontsize=12)
    plt.text(40.5, plt.ylim()[1] * 0.9, '40%', rotation=0, color='b', fontsize=12)
    
    # Labels and title
    plt.title('Distribution of Damage Area', fontsize=14, fontweight='bold')
    plt.xlabel('Percentage of Total Area', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    # Legend
    plt.legend(
        title='Area Range',
        labels=[
            f'<20% (n={counts.get("<20%", 0)})',
            f'20-40% (n={counts.get("20-40%", 0)})',
            f'>40% (n={counts.get(">40%", 0)})'
        ]
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    else:
        plt.show()


def plot_metric_histogram(excel_path, metric_column, bins_edges, bins_labels, 
                         threshold_values, output_path=None):
    """
    Plot histogram for any metric with custom bins.
    
    Args:
        excel_path (str): Path to Excel file
        metric_column (str): Name of the column to plot
        bins_edges (list): Edges for binning (e.g., [0, 0.25, 0.3, np.inf])
        bins_labels (list): Labels for bins (e.g., ['<0.25', '0.25-0.30', '>0.30'])
        threshold_values (list): Values to draw as vertical lines
        output_path (str, optional): Path to save figure
    """
    df = pd.read_excel(excel_path)
    
    if metric_column not in df.columns:
        raise ValueError(f"Excel file must contain '{metric_column}' column")
    
    # Create range categories
    range_column = f'{metric_column}_range'
    df[range_column] = pd.cut(
        df[metric_column],
        bins=bins_edges,
        labels=bins_labels
    )
    
    # Calculate counts
    counts = df[range_column].value_counts()
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(
        df,
        x=metric_column,
        hue=range_column,
        bins=50,
        palette='viridis',
        multiple="stack"
    )
    
    # Add threshold lines
    colors = ['r', 'b', 'g', 'orange']
    for i, threshold in enumerate(threshold_values):
        color = colors[i % len(colors)]
        plt.axvline(threshold, color=color, linestyle='--', linewidth=2)
        plt.text(
            threshold + 0.01,
            plt.ylim()[1] * 0.9,
            f'{threshold}',
            rotation=0,
            color=color,
            fontsize=12
        )
    
    # Labels and title
    plt.title(f'Distribution of {metric_column}', fontsize=14, fontweight='bold')
    plt.xlabel(metric_column, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    # Legend
    legend_labels = [f'{label} (n={counts.get(label, 0)})' for label in bins_labels]
    plt.legend(title=f'{metric_column} Range', labels=legend_labels)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    else:
        plt.show()


def plot_loss_curves(loss_file_paths, labels=None, output_path=None):
    """
    Plot training loss curves from one or more training runs.
    
    Args:
        loss_file_paths (list): List of paths to loss Excel files
        labels (list, optional): Labels for each curve
        output_path (str, optional): Path to save figure
    """
    if not isinstance(loss_file_paths, list):
        loss_file_paths = [loss_file_paths]
    
    if labels is None:
        labels = [f'Run {i+1}' for i in range(len(loss_file_paths))]
    
    plt.figure(figsize=(10, 6))
    
    for i, file_path in enumerate(loss_file_paths):
        df = pd.read_excel(file_path)
        loss_column = df.columns[0]  # Assume first column is loss
        
        plt.plot(df.index, df[loss_column], label=labels[i], linewidth=2)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Curves', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    else:
        plt.show()


def visualize_segmentation_comparison(original_img, segmented_img, title="Segmentation Result"):
    """
    Display original and segmented images side by side.
    
    Args:
        original_img (numpy.ndarray): Original image
        segmented_img (numpy.ndarray): Segmented image
        title (str): Title for the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(segmented_img)
    axes[1].set_title('Segmented Image')
    axes[1].axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    """Example usage."""
    # Example: Plot damage area histogram
    # plot_damage_area_histogram("geometric_metrics_consolidated.xlsx")
    
    # Example: Plot custom metric histogram
    # plot_metric_histogram(
    #     "geometric_metrics_consolidated.xlsx",
    #     metric_column="y dev new",
    #     bins_edges=[0, 0.25, 0.3, np.inf],
    #     bins_labels=['<0.25', '0.25-0.30', '>0.30'],
    #     threshold_values=[0.25, 0.30]
    # )
    
    # Example: Plot loss curves
    # plot_loss_curves(
    #     ["seed_10_loss_values.xlsx", "seed_20_loss_values.xlsx"],
    #     labels=["Seed 10", "Seed 20"]
    # )
    
    print("Visualization utilities loaded.")
    print("Import and use functions as needed.")

