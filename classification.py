import numpy as np
import pandas as pd
from calulation import calculate_totol_accurancy

from data_preprocess import load_data
from kfold_cross_validation import run_KFold

features, labels = load_data('prepared_dataset/climate.csv')

n_splits = 10  # Number of splits for cross-validation
cm_knn_accuracy_list, knn_accuracy_list, cv_knn_accuracy_list, ad_knn_accuracy_list, lmnn_accuracy_list = run_KFold(n_splits, 40, features, labels)

cm_knn_mean, cm_knn_std =calculate_totol_accurancy(cm_knn_accuracy_list)
knn_mean, knn_std =calculate_totol_accurancy(knn_accuracy_list)
cv_knn_mean, cv_knn_std =calculate_totol_accurancy(cv_knn_accuracy_list)
ad_knn_mean, ad_knn_std =calculate_totol_accurancy(ad_knn_accuracy_list)
lmnn_mean, lmnn_std =calculate_totol_accurancy(lmnn_accuracy_list)

print("CM_KNN_total: ", cm_knn_mean, "±", cm_knn_std)
print("KNN_total: ", knn_mean, "±", knn_std)
print("CV_KNN_total: ", cv_knn_mean, "±", cv_knn_std)
print("AD_KNN_total: ", ad_knn_mean, "±", ad_knn_std)
print("LMNN_total: ", lmnn_mean, "±", lmnn_std)