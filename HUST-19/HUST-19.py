import os
import cv2 
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
import skimage
from skimage import measure

target_dir='Demo_Patient_324'
output_dir='Demo_Patient_324_Lung_parenchyma'

model_CT_Valid=load_model("CT_images.model")
model_CT=load_model("CT_Morbidity.model")
model_CF=load_model("CF_Morbidity.model")
model_PLR=joblib.load("HUST-19_Morbidity.model")


def split_target_dir(target_dir, output_dir):
    target_list = [target_dir + os.sep + file for file in os.listdir(target_dir)]
    for target in target_list:
        img_split = split_lung_parenchyma(target, 10999, -96)
        dst = target.replace(target_dir, output_dir)
        dst_dir = os.path.split(dst)[0]
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        cv2.imencode('.jpg', img_split)[1].tofile(dst)
    print(f'Target list done with {len(target_list)} items')


def split_lung_parenchyma(target,size,thr):
    img=cv2.imdecode(np.fromfile(target,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
    try:
        img_thr= cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,size,thr).astype(np.uint8)
    except:
        img_thr= cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,999,thr).astype(np.uint8)
    img_thr=255-img_thr
    img_test=measure.label(img_thr, connectivity = 1)
    props = measure.regionprops(img_test)
    img_test.max()
    areas=[prop.area for prop in props]
    ind_max_area=np.argmax(areas)+1
    del_array = np.zeros(img_test.max()+1)
    del_array[ind_max_area]=1
    del_mask=del_array[img_test]
    img_new = img_thr*del_mask
    mask_fill=fill_water(img_new)
    img_new[mask_fill==1]=255
    img_new = 255-img_new
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(img_new.astype( np.uint8 ))
    labels = np.array(labels, dtype=np.float)
    maxnum = Counter(labels.flatten()).most_common(3)
    maxnum = sorted([x[0] for x in maxnum])
    background = np.zeros_like(labels)
    if len(maxnum) == 1:
        pass
    elif len(maxnum) == 2:
        background[labels == maxnum[1]] = 1
    else:
        background[labels == maxnum[1]] = 1
        background[labels == maxnum[2]] = 1
    img_new[background == 0] = 0
    img_new=cv2.dilate(img_new, np.ones((5,5),np.uint8) , iterations=3)
    img_new = cv2.erode(img_new, np.ones((5, 5), np.uint8), iterations=2)
    img_new = cv2.morphologyEx(img_new, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)),iterations=2)
    img_new = cv2.medianBlur(img_new.astype(np.uint8), 21)
    img_out=img*img_new.astype(bool)
    return img_out


def fill_water(img):
    copyimg = img.copy()
    copyimg.astype(np.float32)

    height, width = img.shape
    img_exp = np.zeros((height + 20, width + 20))
    height_exp, width_exp = img_exp.shape
    img_exp[10:-10, 10:-10] = copyimg

    mask1 = np.zeros([height + 22, width + 22], np.uint8)
    mask2 = mask1.copy()
    mask3 = mask1.copy()
    mask4 = mask1.copy()

    cv2.floodFill(np.float32(img_exp), mask1, (0, 0), 1)
    cv2.floodFill(np.float32(img_exp), mask2, (height_exp - 1, width_exp - 1), 1)
    cv2.floodFill(np.float32(img_exp), mask3, (height_exp - 1, 0), 1)
    cv2.floodFill(np.float32(img_exp), mask4, (0, width_exp - 1), 1)

    mask = mask1 | mask2 | mask3 | mask4

    output = mask[1:-1, 1:-1][10:-10, 10:-10]
    return output


def read_ct_img_bydir(target_dir):
    img=cv2.imdecode(np.fromfile(target_dir,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
    img=cv2.resize(img,(200,200))
    return img


def top_time_series(df,top=10,time_seq=True,del_deficiency=True,invalid_cutoff=0.5):
    df=df[df['NiCT']<=invalid_cutoff]
    df.sort_values('pCT',ascending=0,inplace=True)
    if time_seq:
        df=df.head(top).sort_index()
    else:
        df=df.head(top)
    if len(df)<top:
        print('Patient with not enough CTs')
    return df


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def nom(file):
    w = open('Normalized_'+file, 'w', encoding='utf-8')
    f = open(file, 'r')
    list1 = f.readline().rstrip('\r\n').split('\t')
    w.write('\t'.join(list1) + '\n')
    list2 = f.readline().rstrip('\r\n').split('\t')
    w.write('\t'.join(list2) + '\n')
    list3 = f.readline().rstrip('\r\n').split('\t')
    w.write('\t'.join(list3) + '\n')
    list3 = list3[1:]
    dic={}
    for l in f:
        sp = l.rstrip('\r\n').split('\t')
        dic[sp[0]] = sp
    for key in dic:
        flist = []
        sex = ['Male', 'Female']
        for i, ft in enumerate(dic[key][1:]):
            if i == 0:flist.append(ft)
            if i == 1: flist.append(sex.index(ft))
            if i == 2:
                if is_number(ft):flist.append(float(ft) / 37.2)
                else:flist.append(1)
            if i == 3:
                if ft in ['No underlying disease']:flist.append(0)
                else:flist.append(1)
            if i > 3:
                if '-' in list3[i]:
                    max = float(list3[i].split(' ')[0].split('-')[1])
                    min = float(list3[i].split(' ')[0].split('-')[0])
                elif '<' in list3[i]:
                    max = float(list3[i].split(' ')[0].split('<')[1])
                    min = 0
                ft = ft.replace('>', '').replace('<', '')
                if is_number(ft):flist.append((float(ft) - min) / (max - min))
                else:flist.append(0.5)
        w.write(key  + '\t' + '\t'.join(flist) + '\n')


def X_fromdf(df_top):
    X=np.array([read_ct_img_bydir(file) for file in df_top['File'].tolist()])
    X=X[:,:,:,np.newaxis].transpose(3,1,2,0)[np.newaxis,:,:,:]
    return np.concatenate(X)


split_target_dir(target_dir,output_dir)
img_list=[output_dir+os.sep+file for file in os.listdir(output_dir)]

X_CT_Valid=np.array([read_ct_img_bydir(file) for file in img_list])[:,:,:,np.newaxis]
y_CT_Valid=model_CT_Valid.predict_proba(X_CT_Valid)

df=pd.DataFrame({'File':img_list,'NiCT':y_CT_Valid[:,0],'pCT':y_CT_Valid[:,1],'nCT':y_CT_Valid[:,2]})
df.to_csv('Demo_img_score.txt',sep='\t',index=None)

df_top10=top_time_series(df)
df_top10.to_csv('Demo_top_img_score.txt',sep='\t',index=None)

X_CT=X_fromdf(df_top10)
y_CT=model_CT.predict_proba(X_CT)
file='Demo_Patient_324_CF.txt'
nom(file)
df_CF=pd.read_csv('Normalized_'+file,sep='\t')
X_CF=df_CF.iloc[:,1:].values.astype(float)
y_CF=model_CF.predict(X_CF)

X_PLR=np.hstack([y_CT,y_CF])
y_PLR=model_PLR.predict_proba(X_PLR)

df_PLR=pd.DataFrame({'Control':y_PLR[:,0],'Regular or Mild':y_PLR[:,1],'Critically ill or Severe':y_PLR[:,2]})
df_PLR.to_csv('Demo_result.txt',sep='\t',index=None)
print('Demo done','...','Control',y_PLR[:,0],'Regular or Mild',y_PLR[:,1],'Critically ill or Severe',y_PLR[:,2],sep='\n')
















