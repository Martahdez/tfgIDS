import numpy as np
import pandas as pd

#Preprocess for transfer learning
#FOR UNSW_NB15
dataset = pd.read_csv('train.csv') 
dataset_train = dataset.copy()

dataset_unsw = dataset_train[['proto','service','dur', 'sbytes', 'dbytes', 'label']]

dataset_y = dataset_unsw['label']
dataset_x = dataset_unsw.drop(['label'],1)



dataset_x.to_csv('dataset_prex.csv', index=False)
dataset_y.to_csv('dataset_y.csv',index=False, header='label')