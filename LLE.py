import numpy as np
from sklearn.neighbors import KDTree



def SMO(A, b, knn, num_iterations = 10):
    """
    LLE SMO algorithm to calculate a row vector, w, of weight matrix W
    We are trying to minimize w^T Aw + b^T w, where A is positive definite
    knn contains the only indices that are allowed to be non-zero
    num_iterations is the amount of times we run the algorithm
    """

    # setup
    n = b.shape
    k = len(knn)
    w = np.zeros(n)
    for i in knn:
        w[i] = 1.0 / k
    
    for iteration in range(num_iterations):

        # In every iteration, go over every ordered pair of i, j in knn where i is not j
        for i in knn:
            for j in knn:
                if i==j: continue

                # Want to find optimal difference t to add to w[i] and subtract from w[j]
                # I worked this out on paper
                quadratic_coef = A[i, i] - 2 * A[i, j] + A[j, j]
                linear_coef = 2 * A[i, i] * w[i] + 2 * A[i, j] * w[j] - 2 * A[i, j] * w[i] - 2 * A[j, j] * w[j] + b[i] - b[j]
                t = - linear_coef / (2.0 * quadratic_coef)

                # boundary adjustments
                if t > 0:
                    max_change = min(1 - w[i], w[j])
                    if t > max_change: t = max_change
                elif t < 0:
                    max_change = min(w[i], 1 - w[j])
                    if abs(t) > max_change: t = - max_change

                w[i] += t
                w[j] -= t
    
    return w


def NDR(X, d, k=5):
    """
    X is n by p data matrix
    d is the desired embedding dimension
    k is the number of neighbors in KNN
    """
    n, p = X.shape
    kdtree = KDTree(X)

    W = np.zeros((n, n))
    A = X @ X.T # for optimization later
    for i in range(n): # To find the weight matrix, each row is independently optimized
        knn = kdtree.query([X[i,:]], k=k+1)[1][0] # get k neighbors that aren't itself
        knn = knn[knn != i] # filter out itself
        b = -2 * X @ X[i]      
        W[i, :] = SMO(A, b, knn)
        
    Phi = (np.eye(n) - W.T) @ (np.eye(n) - W)

    eigenvalues, eigenvectors = np.linalg.eig(Phi)
    idx = np.argsort(eigenvalues)
    eigenvectors_sorted = eigenvectors[:, idx]

    return eigenvectors_sorted[:, 1:d+1]




