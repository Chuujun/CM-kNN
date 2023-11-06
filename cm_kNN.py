import numpy as np
import pandas as pd

from calulation import calculate_accuracy, compute_laplacian
from collections import Counter

def calculate_loss(X, Y, L, rho1, rho2, rho3, W):
    R1W = np.linalg.norm(W, ord = 1)
    R2W = np.sum(np.linalg.norm(W, axis=1))
    R3W = np.trace(W.T @ X @ L @ X.T @ W)
    loss = (np.linalg.norm(X.T @ W - Y, ord='fro')) ** 2 + rho1 * R1W + rho2 * R2W + rho3 * R3W

    return loss

def algorithm_1(X, Y, rho1, rho2, rho3, max_iter, tol=1e-5):
    """
    Implements Algorithm 1 to optimize the objective function.

    Parameters:
    - X: Train matrix (n * d) (n train points, d features)
    - Y: Test matrix (d * m) (m test points ,d_features)
    - rho1, rho2, rho3: Regularization parameters
    - L: Graph Laplacian matrix
    - max_iter: Maximum number of iterations
    - tol: Tolerance for convergence

    Returns:
    - W: Weight matrix (n * m) (n test data, m training data)
    """
    n, d = X.shape
    m = Y.shape[1]
    L = compute_laplacian(X)

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

def cm_knn(X, Y, X_label, Y_label, rho1, rho2, rho3, max_iter, tol=1e-5):
    W = algorithm_1(X, Y, rho1, rho2, rho3, max_iter)

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
    accuracy = calculate_accuracy(Y_label, Y_predicted)

    return accuracy
    