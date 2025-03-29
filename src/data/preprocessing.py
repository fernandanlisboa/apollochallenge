import os
import pickle as pkl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(file_path):
    """Loads the data from a pickle file."""
    try:
        with open(file_path, 'rb') as file:
            data = pkl.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error loading the file: {e}")
        return None


def flatten_data(data):
    """Flattens the hierarchical data structure into a pandas DataFrame."""
    try:
        flattened_data = []
        for syndrome_id, subjects in data.items():
            for subject_id, images in subjects.items():
                for image_id, image_data in images.items():
                    record = {
                        "syndrome_id": syndrome_id,
                        "subject_id": subject_id,
                        "image_id": image_id,
                        "embedding": image_data
                    }
                    flattened_data.append(record)
        return pd.DataFrame(flattened_data)
    except Exception as e:
        print(f"Error flattening the data: {e}")
        return pd.DataFrame()


def clean_data(df):
    """Cleans the data by handling missing values and inconsistent embeddings."""
    try:
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            print("Missing values found. Dropping rows with missing values...")
            df = df.dropna()

        # Ensure all embeddings have the same length
        embedding_lengths = df['embedding'].apply(len)
        if embedding_lengths.nunique() > 1:
            print("Inconsistent embedding lengths found. Fixing...")
            expected_length = embedding_lengths.mode()[0]

            def adjust_embedding(embedding):
                if len(embedding) < expected_length:
                    # Pad with zeros
                    return embedding + [0] * (expected_length - len(embedding))
                elif len(embedding) > expected_length:
                    # Truncate to the expected length
                    return embedding[:expected_length]
                return embedding

            df['embedding'] = df['embedding'].apply(adjust_embedding)

        return df
    except Exception as e:
        print(f"Error cleaning the data: {e}")
        return pd.DataFrame()


def normalize_embeddings(df):
    """Normalizes the embeddings in the DataFrame."""
    try:
        embeddings = np.stack(df['embedding'].values)
        norm_embeddings = (embeddings - embeddings.mean(axis=0)) / embeddings.std(axis=0)
        df['embedding'] = list(norm_embeddings)
        return df
    except Exception as e:
        print(f"Error normalizing embeddings: {e}")
        return df



def split_data(df, test_size=0.2):
    """Splits the data into training and test sets."""
    X = np.stack(df["embedding"].values)
    y = df["syndrome_id"].values.ravel()

    # Print shapes antes da divisão
    print(f"Shape of X (embeddings): {X.shape}")
    print(f"Shape of y (labels): {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    print(f"y_train dtype: {y_train.dtype}, example values: {y_train[:10]}")
    print(f"y_test dtype: {y_test.dtype}, example values: {y_test[:10]}")
    # Print shapes após a divisão
    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of y_test: {y_test.shape}")

    return X_train, X_test, y_train, y_test


def save_data(data, output_path):
    """Saves the data to a CSV file."""
    try:
        if isinstance(data, pd.DataFrame):
            data.to_csv(output_path, index=False)
        elif isinstance(data, np.ndarray):
            
            if data.ndim == 1:
                np.savetxt(output_path, data.astype(int), delimiter=",", fmt="%d")  
            else:
                np.savetxt(output_path, data, delimiter=",", fmt="%.6f")  
        elif isinstance(data, (list, pd.Series)):
            np.savetxt(output_path, np.array(data), delimiter=",", fmt="%.6f")
        else:
            raise ValueError("Unsupported data type for saving.")
        print(f"Data successfully saved to '{output_path}'.")
    except Exception as e:
        print(f"Error saving the data: {e}")

def main():
    # Input and output file paths
    filename = "mini_gm_public_v0.1"
    input_path = os.path.join("data", "raw", f"{filename}.p")
    output_dir = os.path.join("data", "preprocessed")
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    data = load_data(input_path)
    if data is None:
        return

    # Flatten data
    df_flattened = flatten_data(data)
    if df_flattened.empty:
        print("Error: Flattened DataFrame is empty.")
        return
    print(f"Shape of flattened DataFrame: {df_flattened.shape}")

    # Clean data
    df_cleaned = clean_data(df_flattened)
    if df_cleaned.empty:
        print("Error: Cleaned DataFrame is empty.")
        return
    print(f"Shape of cleaned DataFrame: {df_cleaned.shape}")

    # Normalize embeddings
    df_normalized = normalize_embeddings(df_cleaned)
    if df_normalized.empty:
        print("Error: Normalized DataFrame is empty.")
        return
    print(f"Shape of normalized DataFrame: {df_normalized.shape}")
    
    # Split data into training and test sets
    x_train, x_test, y_train, y_test = split_data(df_normalized)
    print(f"X_train shape: {x_train.shape}")
    print(f"X_test shape: {x_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Save the split data as CSV files
    save_data(x_train, os.path.join(output_dir, f"X_{filename}_train.csv"))
    save_data(x_test, os.path.join(output_dir, f"X_{filename}_test.csv"))
    save_data(y_train, os.path.join(output_dir, f"y_{filename}_train.csv"))
    save_data(y_test, os.path.join(output_dir, f"y_{filename}_test.csv"))


if __name__ == "__main__":
    main()