import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from collections import Counter

def compute_similarity_matrix(X):
    S = cosine_similarity(X.T)

    return S

def compute_laplacian(X):
    # Compute the similarity_matrix
    S = compute_similarity_matrix(X)
    # Diagonal matrix
    D = np.diag(np.sum(S, axis=1))
    # Laplacian matrix
    L = D - S

    return L

def calculate_distance(x1, x2):
    return np.linalg.norm(x1 - x2)

def calculate_accuracy(Y_actual, Y_predicted):
    return accuracy_score(Y_actual, Y_predicted)

def calculate_RMSE(Y_actual, Y_predicted):
    return mean_squared_error(Y_actual, Y_predicted, squared = False)

def calculate_totol_accurancy(accurancy_list):
    mean = np.mean(accurancy_list)
    std = np.std(accurancy_list, ddof = 1)
    return mean, std
