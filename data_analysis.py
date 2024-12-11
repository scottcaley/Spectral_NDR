import pandas as pd
import numpy as np

import data_creation
import LLE
import Sha_Saul

"""
Results
"""

data = data_creation.create_data(data_creation.spiralling_gaussian_2d, 2, [np.cbrt(x) for x in range(10, 100)])
#data_creation.plot_2d(data)
Y = LLE.NDR(data, 2)
data_creation.plot_2d(Y)
print(Y)