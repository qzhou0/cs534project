import os
import cv2 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from collections import ChainMap
from keras.utils.np_utils import to_categorical
from keras.models import load_model  
from keras.utils import plot_model
from keras import Sequential
from keras import optimizers
from keras.layers import Conv2D,MaxPooling2D,Flatten,Softmax,Activation,Dense
from keras.layers import Input, Dropout, Lambda, Dot, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import Callback,ModelCheckpoint
from sklearn.metrics import roc_curve,roc_auc_score,accuracy_score
from sklearn.linear_model import LogisticRegressionCV 
from sklearn.model_selection import train_val_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


def DNN_Simple(shape):
    X_in = Input(shape=(shape,))
    H2 = Dense(128, activation='relu', kernel_regularizer=l2(5e-4))(X_in)
    H3 = Dropout(0.5)(H2)
    H4 = Dense(64, activation='relu', kernel_regularizer=l2(5e-4))(H3)
    H5 = Dropout(0.2)(H4)
    H6 = Dense(64, activation='relu', kernel_regularizer=l2(5e-4))(H5)
    H7 = Dropout(0.2)(H6)
    H8 = Dense(48, activation='relu', kernel_regularizer=l2(5e-4))(H7)
    H9 = Dropout(0.2)(H8)
    H10 = Dense(16, activation='relu', kernel_regularizer=l2(5e-4))(H9)
    H11 = Dropout(0.5)(H10)
    Y = Dense(3, activation='softmax')(H11)
    model = Model(inputs=X_in, outputs=Y)
    model.compile(optimizer=optimizers.Adam(decay=0.01),loss='categorical_crossentropy'
                  ,metrics=['accuracy']) 
    return model


def Train_fold(i,X_train,y_train,X_val,y_val):
    model=DNN_Simple(127)
    class RocAucEvaluation(Callback):
        def __init__(self, validation_data=()):
            super(Callback, self).__init__()
            self.x_val,self.y_val = validation_data
        def on_epoch_end(self, epoch, log={}):
            y_pred = self.model.predict(self.x_val)
            AUC1 = roc_auc_score(self.y_val[:,0], y_pred[:,0])
            AUC2 = roc_auc_score(self.y_val[:,1], y_pred[:,1])
            AUC3 = roc_auc_score(self.y_val[:,2], y_pred[:,2])
            print('val_AUC Type1 epoch:%d: %.6f' % (epoch+1, AUC1), file=open('Model/DNN_log.txt','a'))
            print('val_AUC Type2 epoch:%d: %.6f' % (epoch+1, AUC2), file=open('Model/DNN_log.txt','a'))
            print('val_AUC Type3 epoch:%d: %.6f' % (epoch+1, AUC3), file=open('Model/DNN_log.txt','a'))
            print(file=open('Model/DNN_log.txt','a'))
    RocAuc = RocAucEvaluation(validation_data=(X_val, y_val))
    checkpoint = ModelCheckpoint(f'Model/DNN_3c_fold_{i+1}_best.model',save_weights_only = False, monitor='val_loss', verbose=1, save_best_only=True,mode='auto',period=1)
    history = model.fit(X_train, y_train, epochs=2000, batch_size=64, class_weight =
                        'auto',validation_data=(X_val, y_val), callbacks=[RocAuc,checkpoint],verbose=1)
    draw_history(history,f'DNN_3c_fold_{i+1}_training_track')


def draw_history(history,title):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title(title+' Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(title+' Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.show()


df=pd.read_csv('CF.txt',sep='\t').drop([0,1]).reset_index(drop=True)
X_list=df.iloc[:,3:]
X=X_list.values.astype(float)
y_list=df.iloc[:,1]
X=df.iloc[:,3:].values.astype(float)
Skf=StratifiedKFold(n_splits=10,shuffle=True)
vals=np.array([])
y_vals=np.array([]).reshape(0,3)
y_val_scores=np.array([]).reshape(0,3)
train, val = Skf.split(X,y_list)[0]
vals=np.append(vals,val)
X_train=X[train]
y_train=to_categorical(y_list.values[train])
X_val=X[val]
y_val=to_categorical(y_list.values[val])
Train_fold(i,X_train,y_train,X_val,y_val)





















