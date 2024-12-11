import numpy as np
import cvxpy as cp
from sklearn.neighbors import KDTree
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA

import LLE
import Laplacian_Eigenmap



def semidef_prog(X, Y, k=5):
    """
    X is the n by p data matrix
    Y is the n by m transformed data matrix (either by LLE or Laplacian Transform)
    k is the number of neighbors
    returns L, a m by m matrix, the next transformation proposed by Sha, Saul
    """

    n, m = Y.shape
    kdtree = KDTree(X)

    # eta is the matrix of eta_{ij}'s
    eta=[[0 for _1 in range(n)] for _2 in range(n)]
    for i in range(n):
        knn = kdtree.query([X[i,:]], k=k)[1][0]
        for j in knn:
            eta[i][j]=1
        
    # Define and solve the CVXPY problem.
    # Create a symmetric matrix variable.
    P = cp.Variable((m,m), symmetric=True)
    # creating the objective function
    objective=0
    for i in range(n):
        #setup s_i
        numer=0
        denom=0
        for j1 in range(n):
            for j2 in range(n):
                if j1!=j2:
                    vecY=Y[j1,:]-Y[j2,:]
                    vecX=X[j1,:]-X[j2,:]
                    numer+=2*eta[i][j1]*eta[i][j2]*cp.quad_form(vecY, P)*(np.linalg.norm(vecX)**2)
                    denom+=2*eta[i][j1]*eta[i][j2]*(np.linalg.norm(vecX)**4)
        s=numer/denom
        for j1 in range(n):
            for j2 in range(n):
                if j1!=j2:
                    vecY=Y[j1,:]-Y[j2,:]
                    vecX=X[j1,:]-X[j2,:]
                    expression=eta[i][j1]*eta[i][j2]*(cp.quad_form(vecY, P)-s*np.linalg.norm(vecX)**2)**2
                    objective+=expression
    # The operator >> denotes matrix inequality.
    constraints = [P >> 0]
    constraints += [cp.trace(P) == 1]
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve()
    
    L = sqrtm(P.value)
    return L



def NDR(X, m, d, use_LLE=True, k=5, variance=0.0):
    """
    X is n by p data matrix
    m is the desired embedding after LLE or Laplacian Eigenmap
    d is the final desired embedding
    use_LLE determines whether LLE or Laplacian Eigenmap is used
    k is number of neighbors
    variance is gaussian variance for Laplacian Eigenmap weight matrix
    """

    Y = LLE.NDR(X, m, k) if use_LLE else Laplacian_Eigenmap.NDR(X, m, k, variance)
    L = semidef_prog(X, Y, k)
    Z = Y @ L # multiply row vectors by symmetric matrix L

    pca = PCA(n_components=d)
    Z_reduced = pca.fit_transform(Z)
    return Z_reduced
