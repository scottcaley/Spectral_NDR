import numpy as np
import math
import matplotlib.pyplot as plt


def plot_2d(X, y=0):
    x1 = X[:, 0]
    x2 = X[:, 1]

    plt.figure()
    plt.scatter(x1, x2)
    plt.show()

def plot_2d(X, y):
    n = X.shape[0]
    x1 = X[:, 0]
    x2 = X[:, 1]

    plt.figure()
    for i in range(n):
        color_val = hex(65536 * math.floor(255 * y[i]) + math.floor(255 * (1 - y[i])))
        color_string = "" + color_val
        color_string = color_string[2:]
        while len(color_string) < 6: color_string = "0" + color_string
        print(color_string)
        plt.scatter(x1[i], x2[i], color='#'+color_string)
    plt.show()