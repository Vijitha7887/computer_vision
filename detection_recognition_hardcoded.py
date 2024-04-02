"""
Ideas for future development:
    don't recoginse overlapping faces
    crop to larger area to see hair
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf

# ################## neural network gender classification ###################
basepath = 'photos'
# output_dir = 'photos_out_3000'
# output_dir = 'photos_out_mac_5'
output_dir = 'photos_out_mac_vgg_3'
# output_dir = 'photos_out_3000_128_3'
# IMG_SIZE = 100
IMG_SIZE = 224
# IMG_SIZE = 160 # for facenet
filenames = os.listdir(basepath)
images = []
grays = []
ages = []
genders = []
how_many = len(filenames)
# # how_many = int(len(filenames)/4)
# # how_many = 10
print('Reading in images...')
for i in range(how_many):
    image = cv2.imread(basepath + '/' + filenames[i])
    # image = cv2.resize(cv2.imread(basepath + '/' + filenames[i]), (IMG_SIZE, IMG_SIZE))
    images.append(image)
    grays.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    metadata = filenames[i].split('_')
    ages.append(int(metadata[0]))
    genders.append(int(metadata[1]))
print('Reading done')

# cv2.imshow('asdf', images[4])

# X = np.array(grays).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
# y = np.array(genders)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# y_test = np.array(genders)
# X_test_grays = np.array(grays).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
# X_test = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

print('Loading model')
# model = tf.keras.models.load_model('my_model_3000')
# model = tf.keras.models.load_model('mac_model_5')
model = tf.keras.models.load_model('mac_model_vgg_3')
# model = tf.keras.models.load_model('my_model_3000_128_3')
print('Model loaded')

# !!! todo: try not transfer learned stuff and age stuff
# we have done it with transfer learning and without it, but
# it was better without because the performance was similar
# and without transfer it was fast. The transfer net is a big
# net.

# ##################### face detection and main loop ############################
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

predicted_genders = []
predicted_binary = []
for imageid, img in enumerate(images):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = grays[imageid]
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        # cropped_face = cv2.resize(gray[y:y+h, x:x+w], (IMG_SIZE, IMG_SIZE))
        cropped_face = cv2.resize(img[y:y+h, x:x+w], (IMG_SIZE, IMG_SIZE))
        # test_data = np.stack([cropped_face], axis=2)
        test_data = cropped_face
        y_proba = model.predict(np.array([test_data])).flatten()
        print()
        print('index', imageid)
        print(y_proba[0])
        print(genders[imageid])
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (x, y-5)
        fontScale = 0.5
        color = (0, 0, 255)
        thickness = 2
        img = cv2.putText(img, '{}%'.format(round(y_proba[0]*100)), org, font, 
                            fontScale, color, thickness, cv2.LINE_AA)
        
        cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 255, 255), 2)
    if len(faces) == 1:
        predicted_genders.append(y_proba[0])
        predicted_binary.append(round(y_proba[0]))
    else:
        predicted_genders.append(np.nan)
        predicted_binary.append(np.nan)
    # cv2.imshow('Original image ' + str(imageid), img)
    cv2.imwrite(output_dir + '/' + str(imageid) + '.jpg', img)

genders = np.array(genders)
predicted_binary = np.array(predicted_binary)
genders = genders[~np.isnan(predicted_binary)]
predicted_binary = predicted_binary[~np.isnan(predicted_binary)]
print('Accuracy score = ', accuracy_score(genders, predicted_binary))
print(confusion_matrix(genders, predicted_binary))
# score = 0.0
# for i in range(len(images)):
#     if predicted_genders[i] is np.nan
# score = np.nanmean((genders[i] - predicted_genders[i])**2)
