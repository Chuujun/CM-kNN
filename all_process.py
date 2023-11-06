import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier as KNN
from collections import Counter

dataset_path = 'prepared_dataset/climate.csv'

# Read the dataset file and convert it to a matrix
df = pd.read_csv(dataset_path)
data_matrix = df.values

# Separating labels and features
features = data_matrix[:, 2 :-1]  # 3-20 columns are features
labels = data_matrix[:, -1]     # last column is label

def calculate_loss(X, Y, L, rho1, rho2, rho3, W):
    R1W = np.linalg.norm(W, ord = 1)
    R2W = np.sum(np.linalg.norm(W, axis=1))
    R3W = np.trace(W.T @ X @ L @ X.T @ W)
    loss = (np.linalg.norm(X.T @ W - Y, ord='fro')) ** 2 + rho1 * R1W + rho2 * R2W + rho3 * R3W

    return loss

def algorithm_1(X, Y, rho1, rho2, rho3, L, max_iter, tol=1e-5):
    """
    Implements Algorithm 1 to optimize the objective function.

    Parameters:
    - X: Train matrix (n * d) means (n train points, d features)
    - Y: Test matrix (d * m) means (m test points ,d_features)
    - rho1, rho2, rho3: Regularization parameters
    - L: Graph Laplacian matrix
    - max_iter: Maximum number of iterations
    - tol: Tolerance for convergence

    Returns:
    - W: Weight matrix (n * m) means (n test data, m training data)
    """
    n, d = X.shape
    m = Y.shape[1]
    # Initialize W randomly
    # W = np.random.randn(n, m)  
    W = np.ones((n,m))

    for iteration in range(max_iter):
        W_old = W.copy()

        # Compute the diagonal matrix D_tilde
        D_tilde = np.zeros((n, n))
        for k in range(n):
            D_tilde[k,k] = 1 / (2 * np.linalg.norm(W[k, :]))

        # For each training data
        for i in range(m):
            # Compute the diagonal matrix D_i
            D_i = np.diag(1.0 / (2.0 * np.abs(W[:, i])))
        
            # Update the weight vector for class i
            A = X @ X.T + rho1 * D_i + rho2 * D_tilde + rho3 * X @ L @ X.T 
            A_inv = np.linalg.inv(A)

            W[:, i] = A_inv @ X @ Y[:, i]

        # Check for convergence
        loss = calculate_loss(X, Y, L, rho1, rho2, rho3, W)
        loss_old = calculate_loss(X, Y, L, rho1, rho2, rho3, W_old)
        diff = np.abs(loss_old - loss)
        print("loss diff: ", diff, " in round ", iteration)
        if diff < tol:
            break
        # print("W in ",iteration, W)

    return W

def calculate_distance(x1, x2):
    return np.linalg.norm(x1 - x2)

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

def calculate_accurancy(Y_actual, Y_predicted):
    correct_count = 0.0
    for idx in range(len(Y_actual)):
        if Y_predicted[idx] == Y_actual[idx]:
            correct_count += 1

    return correct_count / len(Y_actual)

def calculate_RMSE(Y_actual, Y_predicted):
    return mean_squared_error(Y_actual, Y_predicted, squared = False)


# Setting up K-Fold Cross-Validation
n_splits = 10  # Number of splits for cross-validation
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
# KFold
CM_KNN_accurancy_list = []
KNN_accurancy_list = []
for i, (train_index, test_index) in enumerate(kf.split(features)):
    # Splitting the data into training and testing sets
    X, Y = features[train_index], features[test_index]
    Y_dm = np.transpose(Y)
    X_label, Y_label = labels[train_index], labels[test_index]

    # Normalization
    X_normalized = (X - X.mean()) / X.std()
    Y_normalized = (Y - Y.mean()) / Y.std()
    Y_dm_normalized = (Y_dm - Y_dm.mean()) / Y_dm.std()

    L = compute_laplacian(X)

    W = algorithm_1(X_normalized, Y_dm_normalized, 12.0, 5.0, 1.0, L, 1000)
    # Save the W matrix to the file

    # For each test data
    Y_predicted = []
    threshold = 1e-4
    for j in range(W.shape[1]):
        predicted_labels = []
        # For each training data
        for i in range(W.shape[0]):
            if np.abs(W[i][j]) > threshold:
                predicted_labels.append(X_label[i])

        print("K: ",len(predicted_labels))
        y_predicted = Counter(predicted_labels).most_common(1)[0][0]
        Y_predicted.append(y_predicted)


    # CM_KNN test
    CM_KNN_accurancy = accuracy_score(Y_label, Y_predicted)
    print("CM_KNN_accurancy: ", CM_KNN_accurancy)
    CM_KNN_accurancy_list.append(CM_KNN_accurancy)
    # RMSE = calculate_RMSE(Y_label, Y_predicted)
    # print("RMSE: ", RMSE)

    # KNN test
    knn = KNN(n_neighbors=5)
    knn.fit(X_normalized, X_label)
    predicted_labels = knn.predict(Y_normalized)
    KNN_accuracy = accuracy_score(Y_label, predicted_labels)
    print("KNN_accuracy: ", KNN_accuracy)
    KNN_accurancy_list.append(KNN_accuracy)

    if i>1:
        break


# CM_KNN_mean = np.mean(CM_KNN_accurancy_list)
# CM_KNN_std = np.std(CM_KNN_accurancy_list, ddof=1)

# KNN_mean = np.mean(KNN_accurancy_list)
# KNN_std = np.std(KNN_accurancy_list, ddof=1)

# print("CM_KNN_total: ", CM_KNN_mean, "±", CM_KNN_std)
# print("KNN_total: ", KNN_mean, "±", KNN_std)
