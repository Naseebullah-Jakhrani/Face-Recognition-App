import numpy as np
import pandas as pd
import cv2
import pickle

# load All Models
haar = cv2.CascadeClassifier('../FLASK_APP/model/haarcascade_frontalface_default.xml')
#pickle files

mean = pickle.load(open('../FLASK_APP/model/mean_preprocess.pickle', 'rb'))
model_svm = pickle.load(open('../FLASK_APP/model/model_svm.pickle', 'rb'))
model_pca = pickle.load(open('../FLASK_APP/model/pca_50.pickle', 'rb'))

gender_pre = ['Male', 'Female']
font = cv2.FONT_HERSHEY_SIMPLEX


def faceRecognitionPipeline(path, color = 'rgb'):
    if color == 'bgr':
        gray = cv2.cvtColor(path, cv2.COLOR_BGR2GRAY)

    else:
        gray = cv2.cvtColor(path, cv2.COLOR_RGB2GRAY)

    faces = haar.detectMultiScale(gray, 1.5, 3)

    for x,y,w,h in faces:
        cv2.rectangle(path, (x,y), (x+w, y+h), (0,255,0),2)
        roi = gray[y:y+h, x:x+w]
        roi = roi/255
        if roi.shape[1] > 100:
            roi_resize = cv2.resize(roi, (100,100), cv2.INTER_AREA)

        else:
            roi_resize = cv2.resize(roi, (100,100), cv2.INTER_CUBIC)

        roi_reshape = roi_resize.reshape(1, 10000)
        roi_mean = roi_reshape - mean

        eigen_image = model_pca.transform(roi_mean)
        result = model_svm.predict_proba(eigen_image)[0]

        predict = result.argmax()
        score = result[predict]
        text = '%s : %0.2f'%(gender_pre[predict],score)

        cv2.putText(path, text, (x,y), font, 1, (0,255,0),2)
        eig_img = model_pca.inverse_transform(eigen_image)
        
        predictions = [{'roi':roi, 'eig_img':eig_img, 'prediction_name': gender_pre[predict], 'score':score}]
    return path, predictions