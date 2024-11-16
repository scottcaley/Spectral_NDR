import numpy as np

def Locate(X, index, k=5):
    """
    X is n by d data matrix
    index is the index of the desired data point
    k is the number of neighbors
    """
    n = X.shape[0]

    x = X[index]
    distances = []

    for i in range(n):
        if i == index: continue
        distance = np.linalg.norm(x - X[i])
        distances.append((i, distance))

    distances.sort(key=lambda data: data[1])
    return [i for i, _ in distances[:k]]