#Preprocess binary
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


dataset_train = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')
dataset_x = dataset_train.copy()
dataset_tx = dataset_test.copy()


dataset_y = dataset_train['label']
dataset_ty = dataset_test['label']
dataset_x = dataset_x.drop(['label','id','attack_cat'],1)
dataset_tx = dataset_tx.drop(['label','id','attack_cat'],1)

proto_types = np.array(dataset_x['proto'].unique())
service_types = np.array(dataset_x['service'].unique())
state_types = np.array(dataset_x['state'].unique())
#test
proto_typest = np.array(dataset_tx['proto'].unique())
service_typest = np.array(dataset_tx['service'].unique())
state_typest = np.array(dataset_tx['state'].unique())

proto_types_coded = pd.Categorical(dataset_x['proto'], proto_types).codes
service_types_coded = pd.Categorical(dataset_x['service'], service_types).codes
state_types_coded = pd.Categorical(dataset_x['state'], state_types).codes
#test
proto_typest_coded = pd.Categorical(dataset_tx['proto'], proto_typest).codes
service_typest_coded = pd.Categorical(dataset_tx['service'], service_typest).codes
state_typest_coded = pd.Categorical(dataset_tx['state'], state_typest).codes

dataset_x['proto'] = proto_types_coded
dataset_x['service'] = service_types_coded
dataset_x['state'] = state_types_coded
#test
dataset_tx['proto'] = proto_typest_coded
dataset_tx['service'] = service_typest_coded
dataset_tx['state'] = state_typest_coded

dataset_x = dataset_x/dataset_x.max()
dataset_tx = dataset_tx/dataset_tx.max()

x1, x2, y1, y2 = train_test_split(dataset_x, dataset_y, test_size=0.5)
x1_train, x2_train, y1_train, y2_train = train_test_split(x1, y1, test_size=0.5)
x3_train, x4_train, y3_train, y4_train = train_test_split(x2, y2, test_size=0.5)


#Los guardamos
dataset_tx.to_csv('dataset_tx.csv', index=False)
dataset_ty.to_csv('dataset_ty.csv', index=False, header='label')
dataset_x.to_csv('dataset_x.csv', index=False)
dataset_y.to_csv('dataset_y.csv',index=False, header='label')
x1_train.to_csv('x1_train.csv', index=False)
x2_train.to_csv('x2_train.csv', index=False)
y1_train.to_csv('y1_train.csv', index=False, header='label')
y2_train.to_csv('y2_train.csv', index=False, header='label')
x3_train.to_csv('x3_train.csv', index=False)
x4_train.to_csv('x4_train.csv', index=False)
y3_train.to_csv('y3_train.csv', index=False, header='label')
y4_train.to_csv('y4_train.csv', index=False, header='label')
