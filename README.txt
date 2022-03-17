#HUST-19 1.0 (released on Aug 18, 2020)
Hybrid-learning for UnbiaSed predicTion of COVID-19 patients (HUST-19) is a novel framework, which integrated the highly heterogeneous data (CT and CF data) and accurately predicted patients with COVID-19 pneumonia. 
For training the individual image-based model in HUST-19, we manually labelled 19,685 CT slices, including 5705 non-informative CT (NiCT), 4001 positive CT (pCT) and 9979 negative CT (nCT) slices, randomly selected from 61 and 43 patients with and without COVID-19 pneumonia.
For training models to predict morbidity outcomes, we used 197,068 CT slices and 127 types of CF data from 222 control, 438 Type I, and 211 Type II patients in the Cohort 1. We use the Cohort 2 as an independent dataset to test HUST-19, which still exhibits a promising accuracy. In the Cohort 2, there are 91,430 CT slices and CF data from 106 controls, 182 Type I patients, and 63 Type II patients.
For training models to predict mortality outcomes, the Cohort 1 and 2 are merged with 169,933 CT slices and CF data from 662 cured and 57 deceased cases, due to data limitation. All computational models of HUST-19 are made available under a CC BY-NC 4.0 license.

#The description of HUST-19 source code
1. Lung parenchyma.py
The code is used for lung parenchyma extraction.
2. CT_images_CNN.py
The code is used for individual image-based CNN model training.
3. CT_Patient_CNN.py
The code is used for patient-centered CNN model training.
4. CF_DNN.py
The code is used for patient-centered DNN model training.
5. Final_PLR.py
The code is used for the integration of predictions from CT images and CFs.
6. HUST-19.py
The code is the prediction program of HUST-19.

#The Demo input file of HUST-19
We provide a CT folder named "Demo_Patient_324_CT" and a file named "Demo_Patient_324_CF.txt" as input examples for CT and CF, respectively. 

#The models in HUST-19
1. CT_images.model 
The model is used for the CT images prediction.
2. CT_Morbidity.model 
The model is used for COVID-19 patients morbidity outcome prediction based on CT image.
3. CF_Morbidity.model 
The model is used for COVID-19 patients morbidity outcome prediction based on CFs.
4. HUST-19_Morbidity.model 
The model is used for COVID-19 patients morbidity outcome prediction by integrated the CT and CF data.
5. CT_Mortality.model 
The model is used for COVID-19 patients mortality outcome prediction based on CT image.
6. CF_Mortality.model 
The model is used for COVID-19 patients mortality outcome prediction based on CFs.
7. HUST-19_Mortality.model 
The model is used for COVID-19 patients mortality outcome prediction by integrated the CT and CF data.

#The dependencies of HUST-19
To use HUST-19 in your application developments, you must have installed the following dependencies:
Python 3.7
OpenCV-python 3.4.2
Scikit-image 0.15.0
Scikit-learn 0.21.2
Tensorflow 1.13.1
Keras 2.2.4
