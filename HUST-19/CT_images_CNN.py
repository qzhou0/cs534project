import os
import cv2 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import ChainMap
from keras.utils.np_utils import to_categorical
from keras.models import load_model  
from keras.utils.vis_utils import plot_model
from keras import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Softmax,Activation,Dense,Dropout
from keras.callbacks import Callback,ModelCheckpoint
from sklearn.linear_model import LogisticRegressionCV 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from keras import optimizers
from tensorflow.keras.optimizers import Adam

import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

def corp_margin(img2):
    img2=np.asarray(img2)
    (row, col) = img2.shape
    row_top = 0
    raw_down = 0
    col_top = 0
    col_down = 0
    axis1=img2.sum(axis=1)
    axis0=img2.sum(axis=0)
    for r in range(0, row):
        if axis1[r] > 30:
            row_top = r
            break
    for r in range(row - 1, 0, -1):
        if axis1[r] > 30:
            raw_down = r
            break
    for c in range(0, col):
        if axis0[c] > 30:
            col_top = c
            break
    for c in range(col - 1, 0, -1):
        if axis0[c] > 30:
            col_down = c
            break
    a=raw_down+ 1 - row_top-(col_down+ 1-col_top)
    if a>0:
            w=raw_down+ 1-row_top
            col_down=int((col_top+col_down + 1)/2+w/2)
            col_top = col_down-w
            if col_top < 0:
                col_top = 0
                col_down = col_top + w
            elif col_down >= col:
                col_down = col - 1
                col_top = col_down - w
    else:
            w=col_down + 1- col_top
            raw_down = int((row_top + raw_down + 1) / 2 + w/2)
            row_top =  raw_down-w
            if row_top < 0:
                row_top = 0
                raw_down = row_top + w
            elif raw_down >= row:
                raw_down = row - 1
                row_top = raw_down - w
    if row_top==raw_down:
        row_top=0
        raw_down=99
        col_top = 0
        col_down = 99
    new_img = img2[row_top:raw_down + 1, col_top:col_down + 1]
    return new_img


def read_ct_img_bydir(target_dir):
    img=cv2.imdecode(np.fromfile(target_dir,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
    img = corp_margin(img)
    img=cv2.resize(img,(200,200))
    return img


def VGG_Simple():
    model=Sequential()
    model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(200,200,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',kernel_initializer='uniform',activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(32,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(32,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(16,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(16,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3,activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001,decay=0.05),loss='categorical_crossentropy',metrics=['accuracy'])
    #originally optimizers.Adam, and lr= instead of learning_rate = 
    return model


from keras.callbacks import Callback,ModelCheckpoint
class RocAucEvaluation(Callback):
    def __init__(self, validation_data=()):
        super(Callback, self).__init__()
        self.x_val,self.y_val = validation_data
    def on_epoch_end(self, epoch, log={}):
        y_pred = self.model.predict(self.x_val)
        AUC1 = roc_auc_score(self.y_val[:,0], y_pred[:,0])
        AUC2 = roc_auc_score(self.y_val[:,1], y_pred[:,1])
        AUC3 = roc_auc_score(self.y_val[:,2], y_pred[:,2])
        print('val_AUC NiCT epoch:%d: %.6f' % (epoch+1, AUC1))
        print('val_AUC pCT epoch:%d: %.6f' % (epoch+1, AUC2))
        print('val_AUC nCT epoch:%d: %.6f' % (epoch+1, AUC3))
        print()
        self.model.save(f'Model/temp/CT_epoch_{epoch}.model')


target_dir1='NiCT/'
target_dir2='pCT/'
target_dir3='nCT/'
target_list1=[target_dir1+file for file in os.listdir(target_dir1)]
target_list2=[target_dir2+file for file in os.listdir(target_dir2)]
target_list3=[target_dir3+file for file in os.listdir(target_dir3)]

target_list=target_list1+target_list2+target_list3
y_list=to_categorical(np.concatenate(np.array([[0]*len(target_list1),
                                               [1]*len(target_list2),
                                               [2]*len(target_list3)],dtype=object)),3)

X=np.array([read_ct_img_bydir(file) for file in target_list])[:,:,:,np.newaxis]

X_train, X_val, y_train, y_val = train_test_split(X, y_list, test_size=0.1, stratify=y_list)

checkpoint = ModelCheckpoint(f'Model/CT.model',save_weights_only = False, monitor='val_loss', verbose=1,save_best_only=True,mode='auto',save_freq=1)
RocAuc = RocAucEvaluation(validation_data=(X_val,y_val))
model=VGG_Simple()
history = model.fit(X_train, y_train, epochs=300, batch_size=64, class_weight = None, validation_data=(X_val, y_val), callbacks=[checkpoint,RocAuc],verbose=1)
#originally class_weight = 'auto' here


