import numpy as np

"""
Methods for finding bottom eigenvectors.
"""

def Embedding_Transformation(Phi, d):
    """
    Phi is computed from either LLE or Laplacian Eigenmap
    d is the dimension of the desired embedding
    num_zero is the amount of eigenvalues that should be zero
    Return the transformed data
    """
    eigenvalues, eigenvectors = np.linalg.eig(Phi)
    idx = np.argsort(eigenvalues)
    eigenvectors_sorted = eigenvectors[:, idx]

    return eigenvectors_sorted[:, 1:d+1]
