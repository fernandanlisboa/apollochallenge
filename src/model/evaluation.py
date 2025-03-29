"""
Evaluation module for KNN classification.
This module contains functions to visualize and evaluate model performance.
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as KNN


from .metrics import calculate_metrics, calculate_roc


def create_roc_subplot(ax, roc_data, metric_name, class_color_map=None):
    """
    Create a subplot for ROC curves.
    
    Args:
        ax: Matplotlib axis
        roc_data: ROC data for a metric (can be by class or mean)
        metric_name: Name of the metric
        class_color_map: Optional color map for classes
    
    Returns:
        float: Mean AUC for this metric
    """
    # Check if the data is mean ROC or by-class ROC
    if isinstance(roc_data, dict) and "fpr" in roc_data and "tpr" in roc_data and "auc" in roc_data:
        # Mean ROC case
        ax.plot(
            roc_data["fpr"], 
            roc_data["tpr"], 
            label=f"{metric_name} (AUC = {roc_data['auc']:.4f})"
        )
        mean_auc = roc_data["auc"]
    else:
        # By-class ROC case
        aucs = []
        for c, values in sorted(roc_data.items()):
            color = class_color_map.get(c) if class_color_map else None
            ax.plot(
                values["fpr"], 
                values["tpr"], 
                color=color,
                label=f"Class {c} (AUC = {values['auc']:.2f})"
            )
            aucs.append(values["auc"])
        mean_auc = np.mean(aucs)
    
    # Add reference line (random)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    
    # Configure subplot
    ax.set_title(f"{metric_name.capitalize()}\nMean AUC = {mean_auc:.3f}")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.grid(alpha=0.3)
    
    return mean_auc

def plot_roc_comparison(roc_dict, output_dir=None, filename="roc_curve_comparison.png", show=False):
    """
    Plot ROC curves for different metrics in separate subplots.
    
    Args:
        roc_dict: Dictionary with ROC data by metric
        output_dir: Directory to save the plot (optional)
        filename: Name of the file to save (optional)
        show: Whether to display the plot (default: False)
    """
    # Basic configuration
    metric_names = list(roc_dict.keys())
    n_metrics = len(metric_names)
    
    # Create color palette for classes (if data is by class)
    all_classes = set()
    for metric_data in roc_dict.values():
        if not isinstance(metric_data, dict) or ("fpr" not in metric_data):
            # If it's a dictionary by class
            all_classes.update(metric_data.keys())
    
    class_color_map = {c: plt.cm.tab10(i/10) for i, c in enumerate(sorted(all_classes))}
    
    # Create figure and subplots
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5), sharey=True)
    if n_metrics == 1:
        axes = [axes]
    
    # Plot each metric
    for i, metric in enumerate(metric_names):
        create_roc_subplot(axes[i], roc_dict[metric], metric, class_color_map)
        
        # Add labels only where needed
        axes[i].set_xlabel("False Positive Rate")
        if i == 0:
            axes[i].set_ylabel("True Positive Rate")
    
    # Manage legend (if data is by class and there are many classes)
    if len(all_classes) > 10:
        plt.figlegend(loc='lower center', bbox_to_anchor=(0.5, 0), 
                     ncol=min(5, len(all_classes)), fontsize='small')
        plt.subplots_adjust(bottom=0.2)
    else:
        # Legend in each subplot
        for ax in axes:
            ax.legend(loc='lower right', fontsize='small')
    
    plt.suptitle("ROC Curve Comparison", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve comparison plot saved to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()

def plot_mean_roc_comparison(mean_roc_dict, output_dir=None, filename="mean_roc_curve_comparison.png", show=False):
    """
    Plot mean ROC curves for multiple models in the same graph.

    Args:
        mean_roc_dict (dict): Dictionary containing mean ROC curves for each model.
        Example: {"cosine": {"fpr": [...], "tpr": [...], "auc": 0.85}, ...}
        output_dir (str, optional): Directory to save the plot. If None, the plot won't be saved.
        filename (str, optional): Name of the file to save.
        show: Whether to display the plot (default: False)
    """
    plt.figure(figsize=(10, 8))

    # Iterate over models and plot mean ROC curves
    for metric_name, roc_values in mean_roc_dict.items():
        plt.plot(
            roc_values["fpr"], roc_values["tpr"],
            label=f"{metric_name.capitalize()} (AUC = {roc_values['auc']:.4f})"
        )

    # Add diagonal line (random)
    plt.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.50)")

    # Graph settings
    plt.title("Mean ROC Curve Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()

    # Save the plot, if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path)
        print(f"Mean ROC curve comparison plot saved to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, output_dir=None, filename="confusion_matrix.png", show=False):
    """
    Plot confusion matrix for classification results.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names
        output_dir: Directory to save the plot (optional)
        filename: Name of the file to save (optional)
        show: Whether to display the plot (default: False)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def evaluate_models(x_train, y_train, x_test, y_test, model_results_df, distances=["euclidean", "cosine"]):
    """
    Evaluate the best models for each distance metric.
    
    Args:
        x_train (array): Training features
        y_train (array): Training labels
        x_test (array): Testing features
        y_test (array): Testing labels
        model_results_df (DataFrame): DataFrame with cross-validation results
        distances (list): List of distance metrics
        
    Returns:
        tuple: (metrics_results_df, roc_values, mean_roc_dict, best_model)
    """
    metrics_results = {
        "distance_metric": [],
        "dataset": [],
        "accuracy": [],
        "f1_score": [],
        "top3_accuracy": [],
        "n_neighbors": [],
    }
    
    roc_values = {}
    mean_roc_dict = {}
    best_overall_model = None
    best_overall_score = -1
    
    for d in distances:
        best_params = model_results_df.loc[model_results_df["distance"] == d][:1]
        best_distance = best_params["distance"].values[0]
        best_neighbors = int(best_params["n_neighbors"].values[0])
        
        model = KNN(n_neighbors=best_neighbors, metric=best_distance)
        model.fit(x_train, y_train)

        # Save the best overall model
        if best_params["accuracy_mean"].values[0] > best_overall_score:
            best_overall_score = best_params["accuracy_mean"].values[0]
            best_overall_model = model
        
        # Predictions
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
        
        # Probabilities
        y_train_proba = model.predict_proba(x_train)
        y_test_proba = model.predict_proba(x_test)

        # Calculate metrics for training set
        train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba)
        
        metrics_results["n_neighbors"].append(best_neighbors)
        metrics_results["distance_metric"].append(d)
        metrics_results["dataset"].append("train")
        metrics_results["accuracy"].append(train_metrics["accuracy"])
        metrics_results["f1_score"].append(train_metrics["f1_score"])
        metrics_results["top3_accuracy"].append(train_metrics["top3_accuracy"])
        
        # Calculate metrics for test set
        test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)
        
        metrics_results["n_neighbors"].append(best_neighbors)
        metrics_results["distance_metric"].append(d)
        metrics_results["dataset"].append("test")
        metrics_results["accuracy"].append(test_metrics["accuracy"])
        metrics_results["f1_score"].append(test_metrics["f1_score"])
        metrics_results["top3_accuracy"].append(test_metrics["top3_accuracy"])
        
        # Calculate ROC curves
        roc_values[d] = calculate_roc(
            y_true=y_test,
            y_proba=y_test_proba,
            classes=model.classes_,
            by_class=True
        )
        
        mean_roc_dict[d] = calculate_roc(
            y_true=y_test,
            y_proba=y_test_proba,
            classes=model.classes_,
            by_class=False
        )

    metrics_results_df = pd.DataFrame(metrics_results).sort_values(by="accuracy", ascending=False)
    return metrics_results_df, roc_values, mean_roc_dict, best_overall_model