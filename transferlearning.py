#TRANSFER LEARNING MODEL
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
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from keras.utils import plot_model
import time
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
#load dataset
dataset_x = pd.read_csv('x_train.csv')
dataset_y= pd.read_csv('y_train.csv')
dataset_tx = pd.read_csv('x_test.csv')
dataset_ty= pd.read_csv('y_test.csv')
#create model -- binary classification
model = Sequential()
model.add(Dense(20, input_dim=5, activation='relu')) 
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#Load weights if it is need
#model.load_weights('weights/1.h5')
#Compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#Defining the callbacks
callbacks = [EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto'),
 ModelCheckpoint('weights/multi_layer_best_model.h5', monitor='acc', save_best_only=True, verbose=0)]
#Measure the time needed for training
start = time.time()
#Train the model
model.fit(dataset_x,dataset_y,validation_data=(dataset_tx,dataset_ty),callbacks=callbacks,verbose=2,epochs=1000) 
end = time.time()
t = end-start
print(t)
#Evaluate the model
scores = model.evaluate(dataset_tx, dataset_ty) 
print(scores)
#Print the structure of the network
model.summary()
pred1 = model.predict(dataset_tx)
prednew = pred1.round()
y_compare = dataset_ty
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
 # Compute confusion matrix
classes = ['normal','attack']  
cm = confusion_matrix(y_compare, prednew)
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
#ROC CURVE
#definition
def plot_roc(pred,y):
    fpr, tpr, _ = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()
 #ROC curve and AUC
pred = model.predict(dataset_tx)
plot_roc(pred,dataset_ty)
#PRECISIO-RECALL
#Calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(dataset_ty, pred)
# calculate F1 score
f1 = f1_score(dataset_ty, pred.round())
#Precision-recall auc
auc = auc(recall, precision)
print('f1=%.3f auc=%.3f ' % (f1, auc))
# plot no skill
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the precision-recall curve for the model
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.plot(recall, precision, marker='.')
# show the plot
plt.show()
