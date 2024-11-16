import numpy as np
import KNN


def NDR(X, d, k=5):
    """
    X is n by p data matrix
    d is the desired embedding dimension
    k is the number of neighbors in KNN
    """
    n, p = X.shape
    W = np.zeros((n, n))

    for i in range(n):
        """
        To find the weight matrix, each row is independently optimized
        """
        knn = KNN.Locate(X, i, k)

        print(f"i={i}")
        print(knn)

        # We need to optimize W somehow. I'm finding the problem to be a reduction to w^T Aw + b^T w, but A is negative definite.
        # I believe we should use SMO

    Phi = (np.eye(n) - W.T) @ (np.eye(n) - W)

    # PCA on Phi





X = np.array([[1, 2],
              [3, 4],
              [5, 4],
              [3, 1]])
NDR(X, 1, k=2)