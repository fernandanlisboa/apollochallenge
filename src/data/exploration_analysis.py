import os
import pandas as pd
import numpy as np


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


def dataset_statistics(df):
    """Provides basic statistics about the dataset."""
    try:
        print(f"Dataset size: {df.shape}")
        print("\nColumn data types:")
        print(df.dtypes)
        print("\nDescriptive statistics:")
        print(df.describe())
    except Exception as e:
        print(f"Error generating dataset statistics: {e}")


def count_unique_values(df):
    """Counts unique values for key columns."""
    try:
        num_syndromes = df["syndrome_id"].nunique()
        num_subjects = df["subject_id"].nunique()
        num_images = df["image_id"].nunique()

        print(f"\nNumber of unique syndromes: {num_syndromes}")
        print(f"Number of unique subjects: {num_subjects}")
        print(f"Number of unique images: {num_images}")

        return num_syndromes, num_subjects, num_images
    except Exception as e:
        print(f"Error counting unique values: {e}")
        return None, None, None


def group_and_count(df):
    """Groups and counts data by syndrome and subject."""
    try:
        # Count unique subjects per syndrome
        individuals_per_syndrome = df.groupby("syndrome_id")["subject_id"].nunique().rename("subject_count")
        individuals_per_syndrome = individuals_per_syndrome.reset_index()

        # Count images per syndrome
        images_per_syndrome = df.groupby("syndrome_id")["image_id"].count().rename("image_count")
        images_per_syndrome = images_per_syndrome.reset_index()

        print("\nSubjects per syndrome:")
        print(individuals_per_syndrome)

        print("\nImages per syndrome:")
        print(images_per_syndrome)

        return individuals_per_syndrome, images_per_syndrome
    except Exception as e:
        print(f"Error grouping and counting data: {e}")
        return None, None


def detect_outliers(df):
    """Detects outliers in the number of syndromes per individual."""
    try:
        syndromes_per_individual = df.groupby("subject_id")["syndrome_id"].nunique()
        Q1 = syndromes_per_individual.quantile(0.25)
        Q3 = syndromes_per_individual.quantile(0.75)
        IQR = Q3 - Q1
        outliers = syndromes_per_individual[
            (syndromes_per_individual < Q1 - 1.5 * IQR) | (syndromes_per_individual > Q3 + 1.5 * IQR)
        ]

        print(f"\nOutliers detected: {outliers}")
        return outliers
    except Exception as e:
        print(f"Error detecting outliers: {e}")
        return None


def calculate_correlation(df):
    """Calculates the correlation matrix for numerical columns."""
    try:
        correlation_matrix = df.drop(columns=["embedding"]).corr()
        print("\nCorrelation matrix:")
        print(correlation_matrix)
        return correlation_matrix
    except Exception as e:
        print(f"Error calculating correlation: {e}")
        return None


def main():
    # File path for the preprocessed dataset
    filename = "mini_gm_public_v0.1"
    data_path = os.path.join("..", "..", "data", "preprocessed", f"{filename}.csv")

    # Load data
    df = load_data(data_path)
    if df is None:
        return

    # Perform EDA
    dataset_statistics(df)
    count_unique_values(df)
    group_and_count(df)
    detect_outliers(df)
    calculate_correlation(df)


if __name__ == "__main__":
    main()