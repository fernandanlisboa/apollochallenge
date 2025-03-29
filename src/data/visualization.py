import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


def load_data(file_path):
    """Loads the dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"Data successfully loaded from '{file_path}'.")
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error loading the data: {e}")
        return None


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
        plt.savefig(output_path, dpi=300)
        print(f"Bar plot saved to '{output_path}'.")
        
        if show:
            plt.show()
        else:
            plt.close()
    except Exception as e:
        print(f"Error generating bar plot: {e}")


def plot_syndrome_distribution(df, output_dir, show=False):
    """
    Plots the distribution of syndromes in the dataset.
    
    Args:
        df: DataFrame with syndrome data
        output_dir: Directory to save the plot
        show: Whether to display the plot (default: False)
    """
    try:
        syndrome_counts = df["syndrome_id"].value_counts()
        output_path = os.path.join(output_dir, "syndrome_distribution.png")
        save_bar_plot(
            syndrome_counts,
            x_label="Syndrome ID",
            y_label="Count",
            title="Distribution of Syndromes",
            output_path=output_path,
            color="green",
            show=show
        )
    except Exception as e:
        print(f"Error plotting syndrome distribution: {e}")


def plot_individuals_per_syndrome(df, output_dir, show=False):
    """
    Plots the number of individuals per syndrome.
    
    Args:
        df: DataFrame with syndrome and individual data
        output_dir: Directory to save the plot
        show: Whether to display the plot (default: False)
    """
    try:
        individuals_per_syndrome = df.groupby("syndrome_id")["subject_id"].nunique()
        output_path = os.path.join(output_dir, "individuals_per_syndrome.png")
        save_bar_plot(
            individuals_per_syndrome,
            x_label="Syndrome ID",
            y_label="Number of Individuals",
            title="Number of Individuals per Syndrome",
            output_path=output_path,
            color="blue",
            show=show
        )
    except Exception as e:
        print(f"Error plotting individuals per syndrome: {e}")


def plot_images_per_syndrome(df, output_dir, show=False):
    """
    Plots the number of images per syndrome.
    
    Args:
        df: DataFrame with syndrome and image data
        output_dir: Directory to save the plot
        show: Whether to display the plot (default: False)
    """
    try:
        images_per_syndrome = df.groupby("syndrome_id")["image_id"].count()
        output_path = os.path.join(output_dir, "images_per_syndrome.png")
        save_bar_plot(
            images_per_syndrome,
            x_label="Syndrome ID",
            y_label="Number of Images",
            title="Number of Images per Syndrome",
            output_path=output_path,
            color="orange",
            show=show
        )
    except Exception as e:
        print(f"Error plotting images per syndrome: {e}")


def plot_tsne_embeddings(df, output_dir, perplexity=30, n_iter=1000, random_state=42, show=False):
    """
    Applies t-SNE to the embeddings and visualizes them in 2D, saving the figure.
    
    Args:
        df: DataFrame with embeddings and syndrome data
        output_dir: Directory to save the plot
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations for t-SNE
        random_state: Random seed for reproducibility
        show: Whether to display the plot (default: False)
    """
    try:
        # Extract embeddings and syndrome labels
        embeddings = np.stack(df["embedding"].apply(eval))  # Convert stringified lists to numpy arrays
        labels = df["syndrome_id"]

        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_state)
        tsne_results = tsne.fit_transform(embeddings)

        # Create a DataFrame for visualization
        tsne_df = pd.DataFrame(tsne_results, columns=["TSNE-1", "TSNE-2"])
        tsne_df["syndrome_id"] = labels

        # Plot the t-SNE results
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x="TSNE-1", y="TSNE-2", hue="syndrome_id", palette="tab10", data=tsne_df, legend="full", alpha=0.7
        )
        plt.title("t-SNE Visualization of Embeddings")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        # Save the figure
        output_path = os.path.join(output_dir, "tsne_visualization.png")
        plt.savefig(output_path)
        print(f"t-SNE visualization plot saved to '{output_path}'.")
        
        if show:
            plt.show()
        else:
            plt.close()
    except Exception as e:
        print(f"Error visualizing t-SNE embeddings: {e}")


def visualize_tsne(embeddings, labels, output_path=None, show=False):
    """
    Create t-SNE visualization of embeddings.
    
    Args:
        embeddings: The feature vectors
        labels: The syndrome_id labels
        output_path: Path to save the visualization
        show: Whether to display the plot (default: False)
        
    Returns:
        DataFrame with t-SNE results
    """
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'syndrome_id': labels
    })
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=df, x='x', y='y', hue='syndrome_id', palette='viridis')
    plt.title('t-SNE Visualization of Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE visualization saved to '{output_path}'.")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return df


def main():
    # File path for the preprocessed dataset
    filename = "mini_gm_public_v0.1"
    data_path = os.path.join("..", "..", "data", "preprocessed", f"{filename}.csv")

    # Directory to save results
    results_dir = os.path.join("..", "..", "data", "results")
    os.makedirs(results_dir, exist_ok=True)

    # Load data
    df = load_data(data_path)
    if df is None:
        return

    # Plot syndrome distribution
    plot_syndrome_distribution(df, results_dir, show=False)

    # Plot individuals per syndrome
    plot_individuals_per_syndrome(df, results_dir, show=False)

    # Plot images per syndrome
    plot_images_per_syndrome(df, results_dir, show=False)

    # Visualize embeddings using t-SNE
    plot_tsne_embeddings(df, results_dir, show=False)


if __name__ == "__main__":
    main()