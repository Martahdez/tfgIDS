#MULTICLASS CLASSIFICATION MODEL
# MLP in Keras
# Import all the necessary libraries
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import time
from keras.utils import to_categorical
#load dataset
dataset_x = pd.read_csv('x_train.csv')
dataset_y= pd.read_csv('y_train.csv')
dataset_tx = pd.read_csv('x_test.csv')
dataset_ty= pd.read_csv('y_test.csv')
ty = to_categorical(dataset_ty)
y = to_categorical(dataset_y)
#create model 
model = Sequential()
model.add(Dense(20, input_dim=42, activation='relu')) 
model.add(Dense(20, activation='relu'))
model.add(Dense(23, activation='softmax'))
#Load weights if it is need
#model.load_weights('weights/1.h5')
#Compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#Defining the callbacks
callbacks = [EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto'),
 ModelCheckpoint('weights/multi_layer_best_model.h5', monitor='acc', save_best_only=True, verbose=0)]
#Measure the time needed for training
start = time.time()
#Train the model
model.fit(dataset_x,y,validation_data=(dataset_tx,ty),callbacks=callbacks,verbose=2,epochs=1000) 
end = time.time()
t = end-start
print(t)
#Evaluate the model
scores = model.evaluate(dataset_tx, ty) 
print(scores)
#Print the structure of the network
model.summary()
pred1 = model.predict(dataset_tx)
prednew = pred1.round()
y_compare = ty
#CONFUSION MATRIX
#Defining plot_confusion_matrix
def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# Compute confusion matrix multiclass
classes = [0,1,2,3,4,5,6,7,8,9]
cm = confusion_matrix(y_compare, pred)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm, classes)
#Compute normalization matrix confussion
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, classes, title='Normalized confusion matrix')
