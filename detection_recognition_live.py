
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization


# ################## neural network gender classification ###################
# basepath = 'crop_part1/crop_part1'
# IMG_SIZE = 100
IMG_SIZE = 224
# IMG_SIZE = 160 # for facenet
# filenames = os.listdir(basepath)
# images = []
# grays = []
# ages = []
# genders = []
# races = []
# how_many = len(filenames)
# # how_many = int(len(filenames)/4)
# # how_many = 10
# print('Reading in images...')
# for i in range(how_many):
#     # image = cv2.imread(basepath + '/' + filenames[i])
#     image = cv2.resize(cv2.imread(basepath + '/' + filenames[i]), (IMG_SIZE, IMG_SIZE))
#     # images.append(image)
#     grays.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
#     metadata = filenames[i].split('_')
#     ages.append(int(metadata[0]))
#     genders.append(int(metadata[1]))
#     try:
#         races.append(int(metadata[2]))
#     except:
#         print(metadata)
#         races.append('NaN')
#     # print(metadata)
# print('Reading done')    


# ###################### gender: binary classification ######################x
# X = np.array(grays).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
# y = np.array(genders)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    
# def get_model():
#     model = Sequential()
#     model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(BatchNormalization())
#     model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(BatchNormalization())
#     model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(BatchNormalization())
#     model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(BatchNormalization())
#     model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.2))
#     model.add(Flatten())
#     model.add(Dense(256, activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(128, activation='relu'))
#     #model.add(Dropout(0.3))
#     model.add(Dense(1, activation='sigmoid'))
    
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
#     return model

# model = get_model()
    
# model.fit(X_train, y_train, batch_size=50, epochs=1, verbose=1)
# y_proba = model.predict(X_test).flatten()
# y_pred = np.array([round(x) for x in y_proba])
# print('Accuracy score: {:.4f}'.format(accuracy_score(y_test, y_pred)))
# print(confusion_matrix(y_test, y_pred))

############### transfer learning model ########################
print('Loading model')
# model = tf.keras.models.load_model('my_model_3000')
# model = tf.keras.models.load_model('facenet_keras.h5')
# model = tf.keras.models.load_model('my_model_3000_128_3')
model = tf.keras.models.load_model('mac_model_5')
print('Model loaded')

# !!! todo: take average of 5 following predictions - problematic
# !!! todo: try not transfer learned stuff and age stuff
# we have done it with transfer learning and without it, but
# it was better without because the performance was similar
# and without transfer it was fast. The transfer net is a big
# net.

##################### face detection and main loop ############################
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        cropped_face = cv2.resize(gray[y:y+h, x:x+w], (IMG_SIZE, IMG_SIZE))
        # cropped_face = cv2.resize(img[y:y+h, x:x+w], (IMG_SIZE, IMG_SIZE))
        # test_data = np.stack([cropped_face], axis=2)
        test_data = cropped_face
        y_proba = model.predict(np.array([test_data])).flatten()
        # print(y_proba)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (x, y-5)
        fontScale = 0.5
        color = (0, 0, 255)
        thickness = 2
        img = cv2.putText(img, '{}%'.format(round(y_proba[0]*100)), org, font, 
                           fontScale, color, thickness, cv2.LINE_AA)
        
        # cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 255, 255), 2)
    cv2.imshow('Original image', img)    
    
    # cv2.imshow('Grayscale image', gray)
    # for (x, y, w, h) in faces:
    #     cropped_face = gray[y:y+h, x:x+w]
    #     cv2.imshow('Cropped face', cropped_face)
    
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
cap.release()
