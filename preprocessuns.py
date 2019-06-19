#Preprocess and categorization
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#FOR UNSW_NB15
dataset = pd.read_csv('dataset_prex.csv') 
datasetx = dataset.copy()
datasety = pd.read_csv('dataset_y.csv')
#codifcamos dataset_x
proto_tipos = np.array(datasetx['proto'].unique())
prototipos = pd.Categorical(datasetx['proto'], proto_tipos).codes

#update columns
datasetx[datasetx['service'] == 'http'] = 3
datasetx[datasetx['service'] == 'ftp'] = 22
datasetx[datasetx['service'] == 'ftp-data'] = 0
datasetx[datasetx['service'] == 'smtp'] = 15
datasetx[datasetx['service'] == 'pop3'] = 53
datasetx[datasetx['service'] == 'ssh'] = 49
datasetx[datasetx['service'] == 'irc'] = 55
datasetx[datasetx['service'] == 'radius'] = 66
datasetx[datasetx['service'] == '-'] = 67
datasetx[datasetx['service'] == 'dhcp'] = 68
datasetx[datasetx['service'] == 'ssl'] = 69
datasetx[datasetx['service'] == 'snmp'] = 70
datasetx[datasetx['service'] == 'dns'] = 71

datasetx['proto']=prototipos


#normalize
datasetx = datasetx/(381709090)

#after preprocess, divide
x_train, x_test, y_train, y_test = train_test_split(datasetx, datasety, test_size=0.2)
datasetx.to_csv('dataset_x.csv', index=False)
x_train.to_csv('x_train.csv', index=False)
x_test.to_csv('x_test.csv', index=False)
y_train.to_csv('y_train.csv',index=False, header='label')
y_test.to_csv('y_test.csv',index=False, header='label')
