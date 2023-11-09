import numpy as np
import pandas as pd

dataset_paths = [
    'prepared_dataset/climate.csv',
    'prepared_dataset/german.csv'
]

def load_data(dataset_path):
    '''
    Load dataset from dataset_path and extract features and labels.

    Parameters:
    - dataset_path

    Return:
    - features: feature columns of the dataset
    - labels: label columns of the dataset
    '''
    # Read the dataset file and convert it to a matrix
    df = pd.read_csv(dataset_path)
    data_matrix = df.values

    # Separating labels and features
    features = data_matrix[:, 2 :-1]  # 3-20 columns are features
    labels = data_matrix[:, -1]     # last column is label

    return features, labels

def normalize_data(X):
    '''
    Normalize data matrix.

    Parameters:
    - X : matrix 

    Return:
    - normalized_X: normalized matrix
    '''
    X_normalized = (X - X.mean()) / X.std()
    
    return X_normalized

def normalize_data_minmax(X, min_val=1, max_val=2):
    '''
    Normalize data matrix using Min-Max normalization.

    Parameters:
    - X : matrix
    - min_val: minimum value for normalization (default: 0)
    - max_val: maximum value for normalization (default: 1)

    Return:
    - normalized_X: normalized matrix
    '''
    X_normalized = (X - X.min()) / (X.max() - X.min())
    normalized_X = X_normalized * (max_val - min_val) + min_val

    return normalized_X