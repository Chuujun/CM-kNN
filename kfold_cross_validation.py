import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier as KNN

from cm_kNN import cm_knn
from kNN import knn
from data_preprocess import normalize_data

def run_KFold(k, random, features, labels):
    # Setting up K-Fold Cross-Validation

    kf = KFold(n_splits = k, shuffle = True, random_state = random)
    # KFold
    CM_KNN_accuracy_list = []
    KNN_accuracy_list = []
    for i, (train_index, test_index) in enumerate(kf.split(features)):
        # Splitting the data into training and testing sets
        X, Y = features[train_index], features[test_index]
        Y_dm = np.transpose(Y)
        X_label, Y_label = labels[train_index], labels[test_index]

        # Normalization
        X_normalized = normalize_data(X)
        Y_normalized = normalize_data(Y)
        Y_dm_normalized = normalize_data(Y_dm)

        CM_KNN_accuracy = cm_knn(X_normalized, Y_dm_normalized, X_label, Y_label, 12.0, 10.0, 1.0, 1000)
        print("CM_KNN_accuracy: ", CM_KNN_accuracy)
        CM_KNN_accuracy_list.append(CM_KNN_accuracy)

        # KNN test
        KNN_accuracy = knn(X_normalized, Y_normalized, X_label, Y_label, 5)
        print("KNN_accuracy: ", KNN_accuracy)
        KNN_accuracy_list.append(KNN_accuracy)

    return CM_KNN_accuracy_list, KNN_accuracy_list



