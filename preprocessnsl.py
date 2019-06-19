import numpy as np
import pandas as pd#
#script preprocess nsl-kdd
dataset_prey = pd.read_csv('dataset_prey.csv')
datasety = dataset_prey.copy()
dataset_prex = pd.read_csv('dataset_prex.csv')
datasetx = dataset_prex.copy()
#codificamoas
datasety[datasety['label'] != 'normal'] = 1
datasety[datasety['label'] == 'normal'] = 0
datasety.to_csv('dataset_y.csv',index=False, header='label')
#codifcamos dataset_x
service_tipos = np.array(datasetx['service'].unique())
servicetipos = pd.Categorical(datasetx['service'], service_tipos).codes

#update columns - FOR THE COMPARISON
#datasetx[datasetx['proto'] == 'tcp'] = 2
#datasetx[datasetx['proto'] == 'udp'] = 0
#datasetx[datasetx['proto'] == 'icmp'] = 131
#datasetx['service']=servicetipos

#Normalize
datasetx = datasetx/(381709090)
datasetx.to_csv('dataset_x.csv', index=False)