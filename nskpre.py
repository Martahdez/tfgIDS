import numpy as np
import pandas as pd
#Preprocess for transfer learning
#FOR NSL_KDD
dataset = pd.read_csv('kddtrain.csv') # or test
dataset_train = dataset.copy()

dataset_nslkdd = dataset_train[['proto','service','dur', 'sbytes', 'dbytes', 'label']]

dataset_y = dataset_nslkdd['label']
dataset_x = dataset_nslkdd.drop(['label'],1)
dataset_x.to_csv('dataset_prex.csv', index=False)
dataset_y.to_csv('dataset_prey.csv',index=False, header='label')

