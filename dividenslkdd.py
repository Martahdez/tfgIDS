from sklearn.model_selection import train_test_split
import pandas as pd

dataset_x = pd.read_csv('dataset_x.csv')
dataset_y = pd.read_csv('dataset_y.csv')
x1, x2, y1, y2 = train_test_split(dataset_x, dataset_y, test_size=0.2)
x1.to_csv('x1.csv', index=False)
y1.to_csv('y1.csv', index=False, header='label')
x2.to_csv('x2.csv', index=False)
y2.to_csv('y2.csv', index=False, header='label')