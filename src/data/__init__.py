"""
Data processing and visualization package for the Apollo Challenge project.

This package provides functions for data loading, preprocessing, and visualization.
"""

# Import key functions from submodules
from .preprocessing import (
    load_data, 
    flatten_data, 
    clean_data, 
    normalize_embeddings, 
    split_data, 
    save_data
)

from .visualization import (
    plot_syndrome_distribution,
    plot_individuals_per_syndrome,
    plot_images_per_syndrome,
    plot_tsne_embeddings,
    save_bar_plot
)

# Define what should be available when using "from data import *"
__all__ = [
    # Preprocessing functions
    'load_data', 
    'flatten_data',
    'clean_data',
    'normalize_embeddings',
    'split_data',
    'save_data',
    
    # Visualization functions
    'plot_syndrome_distribution',
    'plot_individuals_per_syndrome',
    'plot_images_per_syndrome',
    'plot_tsne_embeddings',
    'save_bar_plot'
]