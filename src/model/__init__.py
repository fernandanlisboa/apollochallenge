"""
KNN Classification Package.
This package implements K-Nearest Neighbors algorithm for classifying syndrome embeddings.
"""

from .metrics import calculate_metrics, calculate_roc
from .evaluation import plot_roc_comparison, plot_mean_roc_comparison
from .classification import load_data, find_optimal_k, save_model

__all__ = [
    'calculate_metrics',
    'calculate_roc',
    'plot_roc_comparison',
    'plot_mean_roc_comparison',
    'load_data',
    'find_optimal_k',
    'save_model'
]