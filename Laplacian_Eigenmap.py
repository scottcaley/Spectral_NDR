import numpy as np
from sklearn.neighbors import KDTree


def NDR(X, d, k=5, variance=0.0):
    """
    X is n by p data matrix
    d is the desired embedding dimension
    k is the number of neighbors in KNN
    variance is a scalar parameter, and its value determines which weight method is chosen
    """
    n, p = X.shape
    kdtree = KDTree(X)

    W = np.zeros((n, n))
    for i in range(n):
        knn = kdtree.query([X[i,:]], k=k+1)[1][0] # get k neighbors that aren't itself
        knn = knn[knn != i] # filter out itself
        for j in knn:
            if (variance <= 0.0):
                W[i, j] = 1.0 / k
            else:
                diff = X[i] - X[j]
                W[i, j] = np.exp(-1.0 * np.dot(diff, diff) / variance)

    D = np.diag(np.sum(W, axis=1))
    D_inv_sqrt = np.diag(1 / np.sqrt(np.diag(D)))
    Phi = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt

    eigenvalues, eigenvectors = np.linalg.eig(Phi)
    idx = np.argsort(eigenvalues)
    eigenvectors_sorted = eigenvectors[:, idx]

    return eigenvectors_sorted[:, 1:d+1]






