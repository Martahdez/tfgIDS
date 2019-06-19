#Preprocess multiclass
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


dataset_train = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')
dataset_x = dataset_train.copy()
dataset_tx = dataset_test.copy()


dataset_y = dataset_train['attack_cat']
dataset_ty = dataset_test['attack_cat']
dataset_x = dataset_x.drop(['label','id'],1)
dataset_tx = dataset_tx.drop(['label','id'],1)

proto_types = np.array(dataset_x['proto'].unique())
service_types = np.array(dataset_x['service'].unique())
state_types = np.array(dataset_x['state'].unique())
attack_types = np.array(dataset_x['attack_cat'].unique())
#test
proto_typest = np.array(dataset_tx['proto'].unique())
service_typest = np.array(dataset_tx['service'].unique())
state_typest = np.array(dataset_tx['state'].unique())
attack_typest = np.array(dataset_tx['attack_cat'].unique())

proto_types_coded = pd.Categorical(dataset_x['proto'], proto_types).codes
service_types_coded = pd.Categorical(dataset_x['service'], service_types).codes
state_types_coded = pd.Categorical(dataset_x['state'], state_types).codes
attack_types_coded = pd.Categorical(dataset_x['attack_cat'], attack_types).codes
#test
proto_typest_coded = pd.Categorical(dataset_tx['proto'], proto_typest).codes
service_typest_coded = pd.Categorical(dataset_tx['service'], service_typest).codes
state_typest_coded = pd.Categorical(dataset_tx['state'], state_typest).codes
attack_typest_coded = pd.Categorical(dataset_tx['attack_cat'], attack_typest).codes

dataset_x['proto'] = proto_types_coded
dataset_x['service'] = service_types_coded
dataset_x['state'] = state_types_coded
dataset_x['attack_cat'] = attack_types_coded
#test
dataset_tx['proto'] = proto_typest_coded
dataset_tx['service'] = service_typest_coded
dataset_tx['state'] = state_typest_coded
dataset_tx['attack_cat'] = attack_typest_coded

dataset_y = dataset_x['attack_cat']
dataset_ty = dataset_tx['attack_cat']
dataset_x = dataset_x.drop(['attack_cat'],1)
dataset_tx = dataset_tx.drop(['attack_cat'],1)

dataset_x = dataset_x/dataset_x.max()


x_train, x_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size=0.2)
x1, x2, y1, y2 = train_test_split(x_train, y_train,test_size=0.5)
x1_train, x2_train, y1_train, y2_train = train_test_split(x1, y1, test_size=0.5)
x3_train, x4_train, y3_train, y4_train = train_test_split(x2, y2, test_size=0.5)

x1t, x2t, y1t, y2t = train_test_split(x_test, y_test, test_size=0.5)
x1_test, x2_test, y1_test, y2_test = train_test_split(x1t, y1t, test_size=0.5)
x3_test, x4_test, y3_test, y4_test = train_test_split(x2t, y2t, test_size=0.5)


#saving them
dataset_x.to_csv('dataset_x.csv', index=False)
dataset_y.to_csv('dataset_y.csv',index=False, header='attack_cat')

x_train.to_csv('x_train.csv', index=False)
x_test.to_csv('x_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False, header='attack_cat')
y_test.to_csv('y_test.csv', index=False, header='attack_cat')

x1_train.to_csv('x1_train.csv', index=False)
x2_train.to_csv('x2_train.csv', index=False)
y1_train.to_csv('y1_train.csv', index=False, header='attack_cat')
y2_train.to_csv('y2_train.csv', index=False, header='attack_cat')
x3_train.to_csv('x3_train.csv', index=False)
x4_train.to_csv('x4_train.csv', index=False)
y3_train.to_csv('y3_train.csv', index=False, header='attack_cat')
y4_train.to_csv('y4_train.csv', index=False, header='attack_cat')
#test
x1_test.to_csv('x1_test.csv', index=False)
x2_test.to_csv('x2_test.csv', index=False)
y1_test.to_csv('y1_test.csv', index=False, header='attack_cat')
y2_test.to_csv('y2_test.csv', index=False, header='attack_cat')
x3_test.to_csv('x3_test.csv', index=False)
x4_test.to_csv('x4_test.csv', index=False)
y3_test.to_csv('y3_test.csv', index=False, header='attack_cat')
y4_test.to_csv('y4_test.csv', index=False, header='attack_cat')