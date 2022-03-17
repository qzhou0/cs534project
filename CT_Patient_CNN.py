import os
import cv2 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import ChainMap
from keras.utils.np_utils import to_categorical
from keras.models import load_model, Model 
from keras.utils import plot_model
from keras import Sequential
from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Conv3D, MaxPooling3D
from keras.layers import Dropout, Input, BatchNormalization
from sklearn.linear_model import LogisticRegressionCV 
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy import sparse
from tqdm import tqdm


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


def read_img_dirs(target_dirs):
    file_list=[]
    for target_dir in target_dirs:
        for root,dirs,files in os.walk(target_dir):
            for file in files:
                file_list.append(os.path.join(root,file).replace("\\",'/'))
    return file_list


def read_ct_img_bydir(target_dir):
    img=cv2.imdecode(np.fromfile(target_dir,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
    img = corp_margin(img)
    img=cv2.resize(img,(200,200))
    return img


def predict_y_scores_3c(target_list,model):
    y_scores=[]
    for target_dir in tqdm(target_list):  
        img=read_ct_img_bydir(target_dir).toarray()
        img=img[np.newaxis,:,:,np.newaxis]
        y_score_3c=model.predict(img)
        y_scores.append(y_score_3c)
    return np.array(y_scores)


def top_time_series(df,top=10,time_seq=True,del_deficiency=True,invalid_cutoff=0.5):
    df=df[df['Invalid_score']<=invalid_cutoff]
    df.sort_values(['Patient','Pos_score'],ascending=[1,0],inplace=True)
    if time_seq:
        df=df.groupby(['Patient']).head(top).sort_index()
    else:
        df=df.groupby(['Patient']).head(top)
    group_size=df.groupby('Patient').size()
    del_group=group_size[group_size<top]
    if del_deficiency:
        df=df[~df['Patient'].isin(del_group.index)]
    return df


def X_y_patient_fromdf(df_top):
    X=[]
    y=[]
    patient_list=[]
    for patient,ds in tqdm(df_top.groupby('Patient')):
        X_patient=np.array([read_ct_img_bydir(file).toarray() for file in ds['File'].tolist()])
        X_patient=X_patient[:,:,:,np.newaxis].transpose(3,1,2,0)
        X.append(X_patient)
        y.append(ds['Type'].iloc[0])
        patient_list.append(patient)
    return patient_list,np.concatenate(X),y


def Train_fold(i,X_train,y_train,X_val,y_val):
    model=VGG_Simple()
    from keras.callbacks import Callback,ModelCheckpoint
    class RocAucEvaluation(Callback):
        def __init__(self, validation_data=()):
            super(Callback, self).__init__()
            self.x_val,self.y_val = validation_data
        def on_epoch_end(self, epoch, log={}):
            y_pred = self.model.predict(self.x_val)
            AUC1 = roc_auc_score(self.y_val[:,0], y_pred[:,0])
            AUC2 = roc_auc_score(self.y_val[:,1], y_pred[:,1])
            print('val_AUC Type1 epoch:%d: %.6f' % (epoch+1, AUC1), file=open('Model/CNN_log.txt','a'))
            print('val_AUC Tyep2 epoch:%d: %.6f' % (epoch+1, AUC2), file=open('Model/CNN_log.txt','a'))
            print(file=open('Model/CNN_log.txt','a'))
            self.model.save(f'Model/temp/CNN_2c_fold_{i+1}_epoch_{epoch}.model')
    RocAuc = RocAucEvaluation(validation_data=(X_val, y_val))
    checkpoint = ModelCheckpoint(f'Model/CNN_2c_fold_{i+1}_best.model',save_weights_only = False ,monitor='val_loss',verbose=1,save_best_only=True,mode='auto',period=1)
    history = model.fit(X_train, y_train, epochs=300, batch_size=64, validation_data=(X_val, y_val),verbose=1,callbacks=[RocAuc,checkpoint],class_weight = 'balanced')
    draw_history(history,f'CNN_2c_fold_{i+1}_training_track')


def VGG_Simple():
    model=Sequential()
    model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(200,200,10),padding='same',activation='relu',kernel_initializer='uniform'))
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
    model.compile(optimizer=optimizers.Adam(lr=0.0007,decay=0.05),loss='categorical_crossentropy',metrics=['accuracy'])
    return model


def draw_history(history,title):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title(title+' Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'validation'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(title+' Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'validation'], loc='upper left')
    plt.show()


from keras.callbacks import Callback
class RocAucEvaluation(Callback):
    def __init__(self, validation_data=()):
        super(Callback, self).__init__()
        self.x_val,self.y_val = validation_data
    def on_epoch_end(self, epoch, log={}):
            y_pred = self.model.predict(self.x_val)
            AUC1 = roc_auc_score(self.y_val[:,0], y_pred[:,0])
            AUC2 = roc_auc_score(self.y_val[:,1], y_pred[:,1])
            AUC3 = roc_auc_score(self.y_val[:,2], y_pred[:,2])
            print('val_all_AUC Control epoch:%d: %.6f' % (epoch+1, AUC1))
            print('val_all_AUC Type 1 epoch:%d: %.6f' % (epoch+1, AUC2))
            print('val_all_AUC Type 2 epoch:%d: %.6f' % (epoch+1, AUC3))


class2type={'Negative': 0,'Mild': 1,'Regular': 1,'Severe': 2,'Critically ill': 2}
type2class={0:'Negative',1: 'Mild + Regular',2:'Severe + Critically ill'}
patient2class={line.strip().split('\t')[0]:line.strip().split('\t')[1] for line in open('Type.txt') if line.strip().split('\t')[1] in class2type}
patient2type={patient:class2type[c]for patient,c in patient2class.items() if c in class2type}
target_list_ref=read_img_dirs(train_dirs)
target_list=[line for line in target_list_ref if line.split('/')[-3] in patient2type]
y_list=[patient2type[line.split('/')[-3]] for line in target_list]
y=to_categorical(y_list)
df=pd.DataFrame({'File':target_list,'Patient_type':[type2class[line] for line in y_list],'Invalid_score':y_scores[:,0],'Pos_score':y_scores[:,1],'Neg_score':y_scores[:,2]})
df.insert(1,'Patient',df['File'].str.split('/').apply(lambda x:x[-3]))
df.to_csv('file_patient_type_score.txt',sep='\t',index=None)
df.to_clipboard(index=None)
df=pd.read_csv('file_patient_score_3c.txt',sep='\t')
df=df[df['Patient'].isin(patient2class)]
df.insert(2,'Class',df['Patient'].map(patient2class))
df.insert(2,'Type',df['Patient'].map(patient2type))
df_top10=top_time_series(df,10)
df_top10.to_clipboard(index=None)
patient_list, X, y_patient = X_y_patient_fromdf(df_top10)
y=to_categorical(y_patient,num_classes=3)
Skf=StratifiedKFold(n_splits=10,shuffle=True)
y_vals=np.array([]).reshape(0,2)
y_vals_scores=np.array([]).reshape(0,2)
val_patients=[]
train, val=Skf.split(patient_list,y_patient)[0]
patient_list_val=(np.array(patient_list)[val]).tolist()
X_train=X[train]
X_val=X[val]
y_train=to_categorical(np.array(y_patient)[train])
y_val=to_categorical(np.array(y_patient)[val])
y_val,y_val_score=Train_fold(i,X_train,y_train,X_val,y_val)





