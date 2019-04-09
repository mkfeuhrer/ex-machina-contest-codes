# from tqdm import tqdm 
# import tensorflow as tf 
import numpy as np
import time
import cv2
import os
import pandas as pd
from keras.applications.resnet50 import ResNet50
from keras.layers import Flatten, Input
from keras.models import Model
from keras.preprocessing import image
from PIL import Image
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.mobilenet import MobileNet
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot

start = time.time()

model = ResNet50(weights='imagenet', pooling=max, include_top = False) 
faceCascade = cv2.CascadeClassifier('./helper.xml')

# GENERATING FEATURES
features = []
test_dir = '../test/'
test_files = os.listdir(test_dir)

i=0

test_final = pd.read_csv('../test_final.csv')
new_test_final = []

for ind,row in test_final.iterrows():
    i = i+1
    if i % 100 == 0: 
        print(i)
    image1 = row['image1']
    image2 = row['image2']
    path = "../test/"
    img1 = cv2.imread(path+image1)
    img2 = cv2.imread(path+image2)

    # img1 = image.load_img(path+image1, target_size=(224, 224))
    # img2 = image.load_img(path+image2, target_size=(224, 224))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    faces1 = faceCascade.detectMultiScale(gray1)
    faces2 = faceCascade.detectMultiScale(gray2)

    if len(faces1)==0:
        crop1=img1
    if len(faces2)==0:
        crop2=img2

    for face in faces1:
        x, y, w, h = face
        crop1 = img1[y:y+h, x:x+w]
        break

    for face in faces2:
        x, y, w, h = face
        crop2 = img2[y:y+h, x:x+w]
        break

    crop1 = cv2.resize(crop1, (224,224))
    crop2 = cv2.resize(crop2, (224,224))

    x1 = image.img_to_array(crop1) 
    x1 = np.expand_dims(x1, axis=0) 
    x1 = preprocess_input(x1) 
    x2 = image.img_to_array(crop2) 
    x2 = np.expand_dims(x2, axis=0) 
    x2 = preprocess_input(x2) 
    features1 = np.asarray(model.predict(x1)).flatten()
    features2 = np.asarray(model.predict(x2)).flatten() 

    # print(features1.shape)
    # print(features2.shape)
    dis = np.linalg.norm(features1-features2)
    # cos_sim = dot(features1, features2)/(norm(features1)*norm(features2))
    # val = 0
    # if cos_sim >= 0.5:
    #     val = 1
    row['target'] = dis
    l1_norm = np.linalg.norm(features1-features2, 1)
    tmp = []
    tmp.append(row['image1'])
    tmp.append(row['image2'])
    tmp.append(row['target'])
    tmp.append(l1_norm)
    new_test_final.append(tmp)

# print(new_test_final)
new_test_final = np.asarray(new_test_final)
res = np.savetxt("../new_test_1.csv",new_test_final, delimiter = ',', fmt="%s")