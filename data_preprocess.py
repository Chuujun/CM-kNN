import numpy as np
import pandas as pd

dataset_paths = [
    'prepared_dataset/climate.csv',
    'prepared_dataset/german.csv'
]

dataset_path = 'prepared_dataset/climate.csv'

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

