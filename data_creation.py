import numpy as np

np.random.seed(0)



def create_data(pdf, d, parameters):
    n = len(parameters)
    X = np.zeros((n, d))
    
    for i in range(n):
        X[i] = pdf(parameters[i])
    return X



def spiralling_gaussian_2d(theta, a=0, b=1, c=1, covariance=np.eye(2)):
    r = a + b * theta**(1.0/c)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    mu = np.array([x, y])
    covariance = 0.2 * np.eye(2)
    return np.random.multivariate_normal(mu, covariance, 1)


