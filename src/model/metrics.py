"""
Metrics module for KNN classification.
This module contains functions to calculate various metrics for model evaluation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score, top_k_accuracy_score
from sklearn.preprocessing import label_binarize

def calculate_mean_roc(y_true, y_proba, classes):
    """
    Calculate the mean ROC curve and AUC across all classes.

    Args:
        y_true (array): True labels.
        y_proba (array): Predicted probabilities for each class.
        classes (array): List of classes.

    Returns:
        dict: Dictionary containing mean FPR, TPR and AUC.
    """
    # Binarize the true labels for multiclass problems
    y_true_bin = label_binarize(y_true, classes=classes)
    
    # Initialize lists to store FPR, TPR and AUC for each class
    fpr_list = []
    tpr_list = []
    auc_list = []
    
    # Calculate ROC curve and AUC for each class
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        auc_score = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(auc_score)
    
    # Calculate the mean FPR and TPR
    all_fpr = np.unique(np.concatenate(fpr_list))
    mean_tpr = np.zeros_like(all_fpr)
    
    for fpr, tpr in zip(fpr_list, tpr_list):
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    
    mean_tpr /= len(classes)
    
    # Calculate the mean AUC
    mean_auc = np.mean(auc_list)
    
    return {"fpr": all_fpr, "tpr": mean_tpr, "auc": mean_auc}

def calculate_roc_by_class(y_true, y_proba, classes):
    """
    Calculate the ROC curve and AUC for each class.

    Args:
        y_true (array): True labels.
        y_proba (array): Predicted probabilities for each class.
        classes (array): List of classes.

    Returns:
        dict: Dictionary containing FPR, TPR and AUC for each class.
    """
    roc_values = {}
    
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve([1 if y == c else 0 for y in y_true], y_proba[:, i])
        auc_score = roc_auc_score([1 if y == c else 0 for y in y_true], y_proba[:, i])
        roc_values[c] = {"fpr": fpr, "tpr": tpr, "auc": auc_score}

    return roc_values

def calculate_roc(y_true, y_proba, classes, by_class=False):
    """
    Calculate ROC curve values.

    Args:
        y_true (array): True labels.
        y_proba (array): Predicted probabilities for each class.
        classes (array): List of classes.
        by_class (bool): If True, return ROC values for each class; otherwise, return mean values.

    Returns:
        dict: Dictionary containing ROC values.
    """
    if by_class:
        return calculate_roc_by_class(y_true, y_proba, classes)
    else:
        return calculate_mean_roc(y_true, y_proba, classes)

def calculate_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate classification metrics.

    Args:
        y_true (array): True labels.
        y_pred (array): Predicted labels.
        y_proba (array, optional): Predicted probabilities for each class.

    Returns:
        dict: Dictionary containing accuracy and F1-score, and optionally top-3 accuracy.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average="macro")
    }
    
    if y_proba is not None:
        metrics["top3_accuracy"] = top_k_accuracy_score(y_true, y_proba, k=3)
    
    return metrics

def plot_correlation_matrix(x_train, y_train, output_dir=None, n_features=20, show=False):
    """
    Plot correlation matrix for the most important features.
    
    Args:
        x_train (array): Training features
        y_train (array): Training labels
        output_dir (str): Directory to save the plot
        n_features (int): Number of features to include in correlation matrix
        show: Whether to display the plot (default: False)
    """
    # Create a DataFrame with the data
    df = pd.DataFrame(x_train)
    df['target'] = y_train
    
    # Calculate correlation with target
    correlations = df.corr()['target'].drop('target')
    
    # Select top features by correlation magnitude
    top_features = correlations.abs().nlargest(n_features).index
    
    # Create correlation matrix for top features
    top_corr = df[list(top_features) + ['target']].corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(top_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(f"Correlation Matrix for Top {n_features} Features")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "correlation_matrix.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Correlation matrix saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()