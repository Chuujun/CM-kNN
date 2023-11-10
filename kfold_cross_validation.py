import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier as KNN

from cm_knn import cm_knn
from data_preprocess import normalize_data_minmax
from baseline import cv_knn, l_knn, ll_knn, ad_knn, knn, lmnn

def run_KFold(k, random, features, labels):
    # Setting up K-Fold Cross-Validation

    kf = KFold(n_splits = k, shuffle = True, random_state = random)
    # KFold
    CM_KNN_accuracy_list = []
    KNN_accuracy_list = []
    L_KNN_accuracy_list = []
    LL_KNN_accuracy_list = []
    CV_KNN_accuracy_list = []
    AD_KNN_accuracy_list = []
    LMNN_accuracy_list = []

    for i, (train_index, test_index) in enumerate(kf.split(features)):
        # Splitting the data into training and testing sets
        X, Y = features[train_index], features[test_index]
        X_label, Y_label = labels[train_index], labels[test_index]

        # Normalization
        X_normalized = normalize_data_minmax(X)
        Y_normalized = normalize_data_minmax(Y)
        Y_dm_normalized = Y_normalized.T

        # CM_KNN test
        CM_KNN_accuracy = cm_knn(X_normalized, Y_dm_normalized, X_label, Y_label, 15, 5, 1, 1000)
        print("CM_KNN_accuracy: ", CM_KNN_accuracy)
        CM_KNN_accuracy_list.append(CM_KNN_accuracy)

        # KNN test
        KNN_accuracy = knn(X_normalized, Y_normalized, X_label, Y_label, 5)
        print("KNN_accuracy: ", KNN_accuracy)
        KNN_accuracy_list.append(KNN_accuracy)

        # CV_KNN test
        CV_KNN_accuracy = cv_knn(X_normalized, Y_normalized, X_label, Y_label)
        print("CV_KNN_accuracy: ", CV_KNN_accuracy)
        CV_KNN_accuracy_list.append(CV_KNN_accuracy)

        # L_KNN test
        L_KNN_accuracy = l_knn(X_normalized, Y_dm_normalized, X_label, Y_label)
        print("L_KNN_accuracy: ", L_KNN_accuracy)
        L_KNN_accuracy_list.append(L_KNN_accuracy)

        # LL_KNN test
        LL_KNN_accuracy = ll_knn(X_normalized, Y_dm_normalized, X_label, Y_label)
        print("LL_KNN_accuracy: ", LL_KNN_accuracy)
        LL_KNN_accuracy_list.append(LL_KNN_accuracy)

        # AD_KNN test
        AD_KNN_accuracy = ad_knn(X_normalized, Y_normalized, X_label, Y_label)
        print("AD_KNN_accuracy: ", AD_KNN_accuracy)
        AD_KNN_accuracy_list.append(AD_KNN_accuracy)

        # LMNN test
        LMNN_accuracy = lmnn(X_normalized, Y_normalized, X_label, Y_label)
        print("LMNN_accuracy: ", LMNN_accuracy)
        LMNN_accuracy_list.append(LMNN_accuracy)

        if (i > 0):
            break

    return CM_KNN_accuracy_list, KNN_accuracy_list, CV_KNN_accuracy_list, L_KNN_accuracy_list, LL_KNN_accuracy_list, AD_KNN_accuracy_list, LMNN_accuracy_list