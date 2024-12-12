import pandas as pd
import numpy as np

import data_creation
import data_vis
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
y = [0, 1/14, 2/14, 3/14, 4/14, 5/14, 6/14, 7/14, 8/14, 9/14, 10/14, 11/14, 12/14, 13/14, 1]
data_vis.plot_2d(data, y)
#Y = Sha_Saul.NDR(data, 2, 2, k=2)
#data_vis.plot_2d(Y, y)
#print(Y)
