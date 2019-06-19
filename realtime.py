#division of the dataset 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

dtotalx = pd.read_csv('x_train.csv')
dataset_x = dtotalx.copy()

dtotaly = pd.read_csv('y_train.csv')
dataset_y = dtotaly.copy()

x1_train, xresto, y1_train, yresto = train_test_split(dataset_x, dataset_y, test_size=0.25)

x1t, x2t, y1t, y2t = train_test_split(xresto, yresto, test_size=0.5)
x5_train, x2_train, y5_train, y2_train = train_test_split(x1t, y1t, test_size=0.5)
x3_train, x4_train, y3_train, y4_train = train_test_split(x2t, y2t, test_size=0.5)




x1_train.to_csv('x1_train.csv', index=False)
x2_train.to_csv('x2_train.csv', index=False)
y1_train.to_csv('y1_train.csv', index=False, header='label')
y2_train.to_csv('y2_train.csv', index=False, header='label')
x3_train.to_csv('x3_train.csv', index=False)
x4_train.to_csv('x4_train.csv', index=False)
y3_train.to_csv('y3_train.csv', index=False, header='label')
y4_train.to_csv('y4_train.csv', index=False, header='label')
x5_train.to_csv('x5_train.csv', index=False)
y5_train.to_csv('y5_train.csv', index=False, header='label')

