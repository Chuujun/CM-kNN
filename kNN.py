import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as KNN

from calulation import calculate_accuracy    


def knn(X, Y, X_label, Y_label, n):
    knn = KNN(n_neighbors=n)
    knn.fit(X, X_label)
    predicted_labels = knn.predict(Y)
    accuracy = calculate_accuracy(Y_label, predicted_labels)
    
    return accuracy