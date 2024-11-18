import numpy as np
import matplotlib.pyplot as plt

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



def plot_2d(data):
    x = data[:, 0]
    y = data[:, 1]

    plt.scatter(x, y)
    plt.grid(True)
    plt.show()



data = create_data(spiralling_gaussian_2d, 2, [np.cbrt(x) for x in range(10, 3000)])
plot_2d(data)

