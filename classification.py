import random
import numpy as np
import pandas as pd
from calulation import calculate_totol_accurancy

from data_preprocess import load_data
from kfold_cross_validation import run_KFold

features, labels = load_data('prepared_dataset/climate.csv')

random_n = random.randint(30, 60)
n_splits = 10  # Number of splits for cross-validation
cm_knn_accuracy_all_list = []
knn_accuracyy_all_list = []
cv_knn_accuracy_all_list = []
l_knn_accuracy_all_list = []
ll_knn_accuracy_all_list = []
ad_knn_accuracy_all_list = []
lmnn_accuracy_all_list = []
for i in range (10):
    print("runing the", i , "th experiment...")
    cm_knn_accuracy_list, knn_accuracy_list, cv_knn_accuracy_list, l_knn_accuracy_list, ll_knn_accuracy_list, ad_knn_accuracy_list, lmnn_accuracy_list = run_KFold(n_splits, random_n, features, labels)
    cm_knn_accuracy_all_list.append(cm_knn_accuracy_list)
    knn_accuracyy_all_list.append(knn_accuracy_list)
    cv_knn_accuracy_all_list.append(cv_knn_accuracy_list)
    l_knn_accuracy_all_list.append(l_knn_accuracy_list)
    ll_knn_accuracy_all_list.append(ll_knn_accuracy_list)
    ad_knn_accuracy_all_list.append(ad_knn_accuracy_list)
    lmnn_accuracy_all_list.append(lmnn_accuracy_list)

cm_knn_mean, cm_knn_std = calculate_totol_accurancy(cm_knn_accuracy_all_list)
knn_mean, knn_std = calculate_totol_accurancy(knn_accuracyy_all_list)
cv_knn_mean, cv_knn_std = calculate_totol_accurancy(cv_knn_accuracy_all_list)
l_knn_mean, l_knn_std = calculate_totol_accurancy(l_knn_accuracy_all_list)
ll_knn_mean, ll_knn_std = calculate_totol_accurancy(ll_knn_accuracy_all_list)
ad_knn_mean, ad_knn_std = calculate_totol_accurancy(ad_knn_accuracy_all_list)
lmnn_mean, lmnn_std = calculate_totol_accurancy(lmnn_accuracy_all_list)

print("CM_KNN_total: ", cm_knn_mean, "±", cm_knn_std)
print("KNN_total: ", knn_mean, "±", knn_std)
print("CV_KNN_total: ", cv_knn_mean, "±", cv_knn_std)
print("L_KNN_total: ", l_knn_mean, "±", l_knn_std)
print("LL_KNN_total: ", ll_knn_mean, "±", ll_knn_std)
print("AD_KNN_total: ", ad_knn_mean, "±", ad_knn_std)
print("LMNN_total: ", lmnn_mean, "±", lmnn_std)