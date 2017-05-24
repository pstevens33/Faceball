import numpy as np
import pandas as pd

X = np.load('../data/X.npy')[:3]
y = np.load('../data/y_wars.npy')[:3]

np.savetxt('../data/X_sample.csv', X)
np.savetxt('../data/y_sample.csv', y)
