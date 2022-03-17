import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegressionCV 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from keras.utils.np_utils import to_categorical


def Train_fold_PLR(i,X_train,y_train,X_test,y_test):
    Clist=[0.001,0.01,0.1,1,10,100,1000,10000]
    clf = LogisticRegressionCV(Cs=Clist,penalty='l2', fit_intercept=True ,cv=10, solver='lbfgs',n_jobs=4
                               , class_weight='balanced',multi_class='multinomial')
    clf.fit(X_train,y_train)
    y_test_scores=clf.predict_proba(X_test)
    y_test_onehot=to_categorical(y_test)
    y_all_scores=clf.predict_proba(X)
    y_all_onehot=to_categorical(y)
    joblib.dump(clf, filename=f'Model/PLR_3c_final_fold_{i}.model')
    print(clf.coef_, file=open(f'Model/plr_weight_fold_{i}.txt','a'))
    print(clf.intercept_,file=open(f'Model/plr_weight_fold_{i}.txt','a'))
    for j in range(3):
        AUC=roc_auc_score(y_test_onehot[:,j],y_test_scores[:,j])
        print(f'Fold {i} AUC Type {j}',AUC)
    for j in range(3):
        AUC=roc_auc_score(y_all_onehot[:,j],y_all_scores[:,j])
        print(f'Fold {i} val_all AUC Type {j}',AUC)
    return y_test_onehot,y_test_scores


df_CNN=pd.read_csv('CNN_patient_score.txt',sep='\t',)
df_DNN=pd.read_csv('DNN_patient_score.txt',sep='\t',)
ind=['Patient','Score0','Score1','Score2']
df=pd.merge(df_CNN[ind],df_DNN[ind],how='inner',on='Patient',suffixes=('_CNN','_DNN'))
df.insert(1,'Type',df['Patient'].map(patient2type))
df.dropna(inplace=True)
df=pd.read_csv('Merged_3c_patient_score.txt',sep='\t',)
X=df.iloc[:,2:].values
y=df['Type'].astype(int).tolist()
df_CTCF=pd.read_csv('CTCF 0321.txt',sep='\t',)
X=df_CTCF.iloc[:,4:].values
y_3c=df_CTCF.iloc[:,1:4].values
y=y_3c.argmax(1)
Skf=StratifiedKFold(n_splits=10,shuffle=True,)
tests=np.array([])
y_tests=np.array([]).reshape(0,3)
y_test_scores=np.array([]).reshape(0,3)
test_patients=[]
for i,(train, test) in enumerate(Skf.split(X,y)):
    tests=np.append(tests,test)
    X_train=X[train]
    X_test=X[test]
    y_train=np.array(y)[train]
    y_test=np.array(y)[test]
    y_test,y_test_score=Train_fold_PLR(i,X_train,y_train,X_test,y_test)

    
    
    
    
    
    
    
    