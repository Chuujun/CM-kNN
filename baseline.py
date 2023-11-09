import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from metric_learn import LMNN
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

from calulation import calculate_accuracy
from sklearn.neighbors import KNeighborsClassifier as KNN

from cm_knn import cm_knn

def knn(X, Y, X_label, Y_label, n):
    knn = KNN(n_neighbors=n)
    knn.fit(X, X_label)
    predicted_labels = knn.predict(Y)
    accuracy = calculate_accuracy(Y_label, predicted_labels)
    
    return accuracy

def cv_knn(X, Y, X_label, Y_label):
    knn = KNeighborsClassifier()
    # 设置CV-KNN K参数范围
    param_grid = {'n_neighbors': range(1, 11)}
    # 进行10倍交叉验证
    kf = KFold(n_splits=10, shuffle=True, random_state=40)  # 创建KFold对象
    # 使用GridSearchCV进行参数搜索和交叉验证
    grid_search = GridSearchCV(knn, param_grid, cv=kf)
    # 拟合CV-KNN模型
    grid_search.fit(X, X_label)
    # 获取最佳k值
    best_k = grid_search.best_params_['n_neighbors']
    # 使用最佳k值创建CV-KNN模型
    cv_knn = KNeighborsClassifier(n_neighbors=best_k)
    cv_knn.fit(X, X_label)
    predicted_labels = cv_knn.predict(Y)
    accuracy = calculate_accuracy(Y_label, predicted_labels)

    return accuracy

def l_knn(X, Y, X_label, Y_label):
    return cm_knn(X, Y, X_label, Y_label, 1, 0, 1, 1000)

def ll_knn(X, Y, X_label, Y_label):
    return cm_knn(X, Y, X_label, Y_label, 1, 1, 0, 1000)

def compute_local_density(X, k):
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    distances, _ = nbrs.kneighbors(X)
    avg_distance = np.mean(distances[:, 1:], axis=1)
    return 1.0 / avg_distance

def ad_knn(X, Y, X_label, Y_label):
    # 创建KNN模型
    k = 20
    knn = KNeighborsClassifier(n_neighbors=k)
    # 计算每个样本的局部密度
    local_density = compute_local_density(X, k)

    # 根据局部密度和预测结果判断是否为异常样本
    is_anomaly = (local_density < np.percentile(local_density, 10))

    # 将异常样本标记为1，正常样本标记为0
    X_label[is_anomaly] = np.ones_like(X_label[is_anomaly])
    # 使用标记后的训练数据进行模型训练
    knn.fit(X, X_label)
    predicted_labels = knn.predict(Y)
    accuracy = calculate_accuracy(Y_label, predicted_labels)

    return accuracy

def lmnn(X, Y, X_label, Y_label):
    k = 5  # 设置K值
    # 创建LMNN模型
    lmnn = LMNN(k=k)
    # 创建KNN模型
    knn = KNeighborsClassifier(n_neighbors=k)
    # 使用LMNN进行距离度量学习
    lmnn.fit(X, X_label)
    # 转换特征空间
    X_transformed = lmnn.transform(X)
    Y_transformed = lmnn.transform(Y)
    knn.fit(X_transformed, X_label)
    predicted_labels = knn.predict(Y_transformed)
    accuracy = calculate_accuracy(Y_label, predicted_labels)

    return accuracy