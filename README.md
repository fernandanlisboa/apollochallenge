# Genetic Syndrome Classification Using KNN

This project implements a machine learning pipeline for classifying genetic syndromes based on facial image embeddings. It uses the K-Nearest Neighbors (KNN) algorithm with different distance metrics (Euclidean and Cosine) to classify syndromes and provides comprehensive evaluation metrics and visualizations.

## Project Structure

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Installation

1.Create a virtual environment and install dependencies:
```
python -m venv venv source venv/bin/activate # On Windows: venv\Scripts\activate pip install -r requirements.txt
```

## Data

The project uses a dataset of facial image embeddings from individuals with various genetic syndromes. The raw data is stored in a pickle format with the following structure:
- Image ID
- Syndrome ID
- Subject ID
- Embedding vector (facial features extracted from images)

## Usage

### Running the Complete Pipeline

To run the full classification pipeline:

```bash
python main.py
```
This will:
1. Load and preprocess the data
2. Perform exploratory data analysis
3. Generate visualizations (t-SNE, syndrome distributions)
4. Train KNN models with different distance metrics
5. Evaluate the models using 10-fold cross-validation
6. Generate performance metrics and visualizations
7. Save the best model and results

### Individual Modules

You can also use individual components of the pipeline:

#### Data Preprocessing
```python
from src.data.preprocessing import load_data, flatten_data, clean_data, normalize_embeddings, split_data

# Load and preprocess data
raw_data = load_data(input_path)
df_flattened = flatten_data(raw_data)
df_cleaned = clean_data(df_flattened)
df_normalized = normalize_embeddings(df_cleaned)

# Split data
X_train, X_test, y_train, y_test = split_data(df_normalized)
```

#### Data Visualization
```python
from src.data.visualization import plot_syndrome_distribution, plot_tsne_embeddings

# Visualize syndrome distribution
plot_syndrome_distribution(df, results_dir)

# Visualize embeddings using t-SNE
plot_tsne_embeddings(df, results_dir)
```

#### Model Training and Evaluation
```python
from src.model.classification import find_optimal_k
from src.model.evaluation import evaluate_models, plot_roc_comparison

# Find optimal k using cross-validation
model_results_df = find_optimal_k(X_train, y_train, distances=["euclidean", "cosine"])

# Evaluate models
metrics_df, roc_values, mean_roc_dict, best_model = evaluate_models(
    X_train, y_train, X_test, y_test, model_results_df
)

# Plot ROC curves
plot_roc_comparison(roc_values, output_dir=results_dir)
```

#### Features
- Data Processing: Load, flatten, clean, and normalize data from hierarchical structure
- Exploratory Data Analysis: Generate statistics and visualizations of dataset characteristics
- t-SNE Visualization: Reduce dimensionality and visualize embeddings by syndrome
- KNN Classification: Implement KNN with both Euclidean and Cosine distance metrics
- Hyperparameter Optimization: Find optimal k value using 10-fold cross-validation
- Comprehensive Evaluation: Calculate and visualize various performance metrics:
  - Accuracy
    - F1-Score
    - Top-k Accuracy
    - ROC curves and AUC values

#### Key Results

The project demonstrates that:

1. Cosine distance generally outperforms Euclidean distance for facial embedding classification
2. The optimal number of neighbors (k) varies between different distance metrics
3. Some syndromes are more easily classified than others
4. t-SNE visualization reveals natural clustering of syndromes in the embedding space