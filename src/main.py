"""
Main execution script for the syndrome classification pipeline.
This script runs the entire workflow from data loading to model evaluation.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE

# Ensure project root is in path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import modules
from data.preprocessing import load_data, flatten_data, clean_data, normalize_embeddings, split_data
from data.visualization import visualize_tsne, plot_tsne_embeddings, plot_syndrome_distribution, plot_individuals_per_syndrome, plot_images_per_syndrome
from model.classification import find_optimal_k, save_model
from model.evaluation import plot_roc_comparison, plot_mean_roc_comparison, evaluate_models
from model.metrics import calculate_metrics, calculate_roc, plot_correlation_matrix


def save_bar_plot(data, x_label, y_label, title, output_path, color="blue", show=False):
    """
    Generates and saves a bar plot.
    
    Args:
        data: Data to plot
        x_label: X-axis label
        y_label: Y-axis label
        title: Plot title
        output_path: Path to save the plot
        color: Bar color
        show: Whether to display the plot (default: False)
    """
    try:
        plt.figure(figsize=(12, 6))
        sns.barplot(x=data.index, y=data.values, color=color)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Garante que o diretório existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        plt.savefig(output_path, dpi=300)
        print(f"Bar plot saved to '{output_path}'.")
        
        if show:
            plt.show()
        else:
            plt.close()
    except Exception as e:
        print(f"Error generating bar plot: {e}")


def analyze_and_visualize_data(df, results_dir, show_plots=False):
    """
    Perform exploratory data analysis and generate visualizations.
    
    Args:
        df: DataFrame with the processed dataset
        results_dir: Directory to save results
        show_plots: Whether to display plots (default: False)
    """
    print("\n===== EXPLORATORY DATA ANALYSIS =====")
    
    # Generate basic statistics
    syndrome_counts = df.groupby('syndrome_id').size()
    subjects_per_syndrome = df.groupby('syndrome_id')['subject_id'].nunique()
    
    print(f"Dataset Summary:")
    print(f"- Total samples: {len(df)}")
    print(f"- Number of unique syndromes: {len(syndrome_counts)}")
    print(f"- Number of unique subjects: {df['subject_id'].nunique()}")
    
    print("\nTop 5 most common syndromes:")
    for i, (syndrome, count) in enumerate(syndrome_counts.sort_values(ascending=False)[:5].items()):
        print(f"  {i+1}. Syndrome {syndrome}: {count} samples")
    
    print("\nBottom 5 least common syndromes:")
    for i, (syndrome, count) in enumerate(syndrome_counts.sort_values()[:5].items()):
        print(f"  {i+1}. Syndrome {syndrome}: {count} samples")
    
    # Plot syndrome distribution
    print("\nGenerating syndrome distribution plot...")
    try:
        output_path = os.path.join(results_dir, "syndrome_distribution.png")
        save_bar_plot(
            syndrome_counts.sort_values(ascending=False),
            x_label="Syndrome ID",
            y_label="Count",
            title="Distribution of Syndromes",
            output_path=output_path,
            color="green",
            show=show_plots
        )
    except Exception as e:
        print(f"Error plotting syndrome distribution: {e}")
    
    # Plot individuals per syndrome
    print("\nGenerating individuals per syndrome plot...")
    try:
        output_path = os.path.join(results_dir, "individuals_per_syndrome.png")
        save_bar_plot(
            subjects_per_syndrome.sort_values(ascending=False),
            x_label="Syndrome ID",
            y_label="Number of Individuals",
            title="Number of Individuals per Syndrome",
            output_path=output_path,
            color="blue",
            show=show_plots
        )
    except Exception as e:
        print(f"Error plotting individuals per syndrome: {e}")
    
    # Plot images per syndrome if image_id exists
    if 'image_id' in df.columns:
        print("\nGenerating images per syndrome plot...")
        try:
            images_per_syndrome = df.groupby("syndrome_id")["image_id"].count()
            output_path = os.path.join(results_dir, "images_per_syndrome.png")
            save_bar_plot(
                images_per_syndrome.sort_values(ascending=False),
                x_label="Syndrome ID",
                y_label="Number of Images",
                title="Number of Images per Syndrome",
                output_path=output_path,
                color="orange",
                show=show_plots
            )
        except Exception as e:
            print(f"Error plotting images per syndrome: {e}")
    
    # Generate t-SNE visualization
    print("\nGenerating t-SNE visualization...")
    try:
        # Check if embeddings are in a format that needs conversion
        if isinstance(df['embedding'].iloc[0], str):
            # Convert string representations to arrays
            embeddings = np.stack(df["embedding"].apply(eval))
        else:
            # Already in array format
            embeddings = np.stack(df["embedding"].tolist())
            
        labels = df["syndrome_id"].values
        output_path = os.path.join(results_dir, "tsne_visualization.png")
        
        # Apply t-SNE and create visualization
        
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Create a DataFrame for visualization
        tsne_df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'syndrome_id': labels
        })
        
        plt.figure(figsize=(12, 10))
        sns.scatterplot(data=tsne_df, x='x', y='y', hue='syndrome_id', palette='viridis')
        plt.title('t-SNE Visualization of Embeddings')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE visualization saved to '{output_path}'.")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
            
    except Exception as e:
        print(f"Error generating t-SNE visualization: {e}")


def run_pipeline():
    """Run the entire data processing and modeling pipeline."""
    print("======== SYNDROME CLASSIFICATION PIPELINE ========")
    
    # Configuration
    filename = "mini_gm_public_v0.1"
    input_path = os.path.join(project_root, "..", "data", "raw", f"{filename}.p")
    output_dir = os.path.join(project_root, "..", "data", "preprocessed")
    results_dir = os.path.join(project_root, "..", "data", "results")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Working directory: {os.getcwd()}")
    print(f"Results will be saved to: {results_dir}")
    
    # 1. Data Processing
    print("\n===== DATA PROCESSING =====")
    print("Loading and preprocessing data...")
    raw_data = load_data(input_path)
    df_flattened = flatten_data(raw_data)
    df_cleaned = clean_data(df_flattened)
    df_normalized = normalize_embeddings(df_cleaned)
    
    # Save preprocessed data
    preprocessed_path = os.path.join(output_dir, f"{filename}_preprocessed.csv")
    df_normalized.to_csv(preprocessed_path, index=False)
    print(f"Preprocessed data saved to: {preprocessed_path}")
    
    # 2. Data Splitting
    X_train, X_test, y_train, y_test = split_data(df_normalized)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # 3. Exploratory Data Analysis and Visualization
    analyze_and_visualize_data(df_normalized, results_dir, show_plots=False)
    
    # 5. Classification with KNN
    print("\n===== KNN CLASSIFICATION =====")
    print("Finding optimal k for KNN classification...")
    distances = ["euclidean", "cosine"]
    model_results_df = find_optimal_k(X_train, y_train, distances)
    
    # Save cross-validation results
    cv_results_path = os.path.join(results_dir, "knn_cv_results.csv")
    model_results_df.to_csv(cv_results_path, index=False)
    print(f"Cross-validation results saved to: {cv_results_path}")
    
    # Display best parameters for each distance metric
    print("\nBest parameters by distance metric:")
    for dist in distances:
        best = model_results_df[model_results_df['distance'] == dist].iloc[0]
        print(f"- {dist.capitalize()}: k={int(best['n_neighbors'])}, accuracy={best['accuracy_mean']:.4f}")
    
    # 6. Evaluate best models
    print("\n===== MODEL EVALUATION =====")
    print("Evaluating final models...")
    metrics_df, roc_values, mean_roc_dict, best_model = evaluate_models(
        X_train, y_train, X_test, y_test, model_results_df, distances
    )
    
    # Save evaluation metrics
    metrics_path = os.path.join(results_dir, "classification_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Evaluation metrics saved to: {metrics_path}")
    
    # Display metrics summary
    print("\nPerformance metrics summary:")
    test_metrics = metrics_df[metrics_df['dataset'] == 'test']
    for i, row in test_metrics.iterrows():
        print(f"- {row['distance_metric'].capitalize()} (k={int(row['n_neighbors'])}): "
              f"Accuracy={row['accuracy']:.4f}, F1={row['f1_score']:.4f}, "
              f"Top-3 Accuracy={row['top3_accuracy']:.4f}")
    

    # 7. Create evaluation plots
    print("\n===== GENERATING EVALUATION PLOTS =====")
    print("Creating ROC curves...")
    plot_roc_comparison(roc_values, output_dir=results_dir)
    plot_mean_roc_comparison(mean_roc_dict, output_dir=results_dir)

    # Adicionar matriz de correlação
    print("Generating correlation matrix...")
    plot_correlation_matrix(X_train, y_train, output_dir=results_dir)
    # 8. Save the best model
    print("\n===== SAVING BEST MODEL =====")
    best_model_path = os.path.join(results_dir, "best_knn_model.pkl")
    save_model(best_model, results_dir)
    print(f"Best model saved to: {best_model_path}")
    
    print("\n===== PIPELINE COMPLETED SUCCESSFULLY =====")
    print(f"All results are available in: {results_dir}")


if __name__ == "__main__":
    run_pipeline()