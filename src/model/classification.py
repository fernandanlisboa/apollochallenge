"""
Classification module for KNN algorithm.
This module implements the K-Nearest Neighbors (KNN) algorithm for classifying embeddings.
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import cross_val_score, train_test_split

# Import our custom modules
from .metrics import plot_correlation_matrix
from .evaluation import (
    plot_roc_comparison, 
    plot_mean_roc_comparison, 
    plot_confusion_matrix,
    evaluate_models
)

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

# Now use paths relative to project root
data_dir = os.path.join(project_root, "data", "preprocessed")

def load_data(filename, data_dir):
    """
    Load training and testing data.
    
    Args:
        filename (str): Base filename for the dataset
        data_dir (str): Directory containing the data files
        
    Returns:
        tuple: (x_train, x_test, y_train, y_test)
    """
    x_train_path = os.path.join(data_dir, f"X_{filename}_train.csv")
    x_test_path = os.path.join(data_dir, f"X_{filename}_test.csv")
    y_train_path = os.path.join(data_dir, f"y_{filename}_train.csv")
    y_test_path = os.path.join(data_dir, f"y_{filename}_test.csv")

    x_train = np.loadtxt(x_train_path, delimiter=",")
    x_test = np.loadtxt(x_test_path, delimiter=",")
    y_train = np.loadtxt(y_train_path, delimiter=",", dtype=int)
    y_test = np.loadtxt(y_test_path, delimiter=",", dtype=int)
    
    return x_train, x_test, y_train, y_test

def find_optimal_k(x_train, y_train, distances=["euclidean", "cosine"], k_range=range(1, 16), cv=10):
    """
    Find the optimal value of k for KNN using cross-validation.
    
    Args:
        x_train (array): Training features
        y_train (array): Training labels
        distances (list): List of distance metrics to compare
        k_range (range): Range of k values to test
        cv (int): Number of cross-validation folds
        
    Returns:
        pd.DataFrame: DataFrame with cross-validation results
    """
    model_results = {
        "distance": [],
        "n_neighbors": [],
        "accuracy_mean": []
    }
    
    for d in distances:
        print(f"Training with {d} distance")
        for n in k_range:
            model = KNN(n_neighbors=n, metric=d)
            acc_mean = cross_val_score(model, x_train, y_train, cv=cv, scoring="accuracy")
            model_results["distance"].append(d)
            model_results["n_neighbors"].append(n) 
            model_results["accuracy_mean"].append(float(acc_mean.mean()))
    
    model_results_df = pd.DataFrame(model_results)
    model_results_df = model_results_df.sort_values(by="accuracy_mean", ascending=False)
    
    return model_results_df

def save_model(model, output_dir, filename="best_knn_model.pkl"):
    """
    Save trained model to disk.
    
    Args:
        model: Trained model to save
        output_dir (str): Directory to save the model
        filename (str): Name of the file to save
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"Model saved to {output_path}")

def main():
    """Main execution function."""
    # Configuration
    filename = "mini_gm_public_v0.1"
    output_dir = os.path.join("..", "..", "data", "results")
    distances = ["euclidean", "cosine"]
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    x_train, x_test, y_train, y_test = load_data(filename, data_dir)
    print(f"x_train shape: {x_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Find optimal k using cross-validation
    model_results_df = find_optimal_k(x_train, y_train, distances)
    print("\nCross-validation results:")
    print(model_results_df.head(10))
    
    # Save cross-validation results
    model_results_df.to_csv(os.path.join(output_dir, "knn_results_cv.csv"), index=False)
    print("Cross-validation results saved to 'knn_results_cv.csv'")
    
    # Evaluate models with the best parameters
    metrics_results_df, roc_values, mean_roc_dict, best_model = evaluate_models(
        x_train, y_train, x_test, y_test, model_results_df, distances
    )
    
    # Print and save metrics results
    print("\nMetrics results:")
    print(metrics_results_df)
    metrics_results_df.to_csv(os.path.join(output_dir, "metrics_results.csv"), index=False)
    print("Metrics results saved to 'metrics_results.csv'")
    
    # Plot ROC curves
    plot_roc_comparison(roc_values, output_dir=output_dir, filename="roc_curve_comparison.png")
    plot_mean_roc_comparison(mean_roc_dict, output_dir=output_dir)
    
    # Plot confusion matrix for the best model
    y_test_pred = best_model.predict(x_test)
    plot_confusion_matrix(
        y_test, y_test_pred, 
        classes=best_model.classes_,
        output_dir=output_dir
    )
    
    # Plot feature correlation matrix
    plot_correlation_matrix(x_train, y_train, output_dir=output_dir)
    
    # Save the best model
    save_model(best_model, output_dir)
    
    # Print summary
    print("\nProcessing completed. Results are available in:", output_dir)

if __name__ == "__main__":
    main()