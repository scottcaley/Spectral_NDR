import pandas as pd
import numpy as np

import data_creation
import LLE
import Laplacian_Eigenmap
import Sha_Saul

"""
Results
"""

data = np.asarray([[-1, 1],
                   [-0.75, 1],
                   [-0.5, 1],
                   [-0.25, 1],
                   [0, 1],
                   [0.259, 0.966],
                   [0.5, 0.866],
                   [0.707, 0.707],
                   [0.866, 0.5],
                   [0.966, 0.259],
                   [1, 0],
                   [1, -0.25],
                   [1, -0.5],
                   [1, -0.75],
                   [1, -1]])
#data_creation.plot_2d(data)
Y = Sha_Saul.NDR(data, 2, 2, k=2)
#data_creation.plot_2d(Y)
print(Y)