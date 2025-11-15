"""
Post-processing module for rice grain segmentation analysis.

This module provides tools for:
- Color normalization of segmentation predictions
- Geometric and quality metrics calculation
- Visualization and analysis
"""

from .utils import (
    quantize_colors,
    quantize_to_fixed_palette,
    find_unique_colors,
    replace_color_exact,
    print_colors_for_inspection
)

__all__ = [
    'quantize_colors',
    'quantize_to_fixed_palette',
    'find_unique_colors',
    'replace_color_exact',
    'print_colors_for_inspection',
]

