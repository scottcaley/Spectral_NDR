import numpy as np
import KNN
import PCA



def NDR(X, d, k=5, variance=0.0):
    """
    X is n by p data matrix
    d is the desired embedding dimension
    k is the number of neighbors in KNN
    variance is a scalar parameter, and its value determines which weight method is chosen
    """
    n, p = X.shape

    W = np.zeros((n, n))
    for i in range(n):
        knn_i = KNN.Locate(X, i, k)
        for j in knn_i:
            if (variance <= 0.0):
                W[i, j] = 1.0 / k
            else:
                diff = X[i] - X[j]
                W[i, j] = np.exp(-1.0 * np.dot(diff, diff) / variance)
    W = 0.5 * (W + W.T) # iffy on this

    D = np.diag(np.sum(W, axis=1))
    D_inv_sqrt = np.diag(1 / np.sqrt(np.diag(D)))
    Phi = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt

    return PCA.Embedding_Transformation(Phi, d)






