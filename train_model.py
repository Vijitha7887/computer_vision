"""
todo: implement and plot training history, show overfitting for dnn
"""

import numpy as np
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import models, Sequential
from keras import layers
import cv2
import os
import pickle

def train_vgg(data_path, model_path, epochs):
    vgg = VGG16(include_top=False, pooling='avg', weights='imagenet', input_shape=(178, 218, 3))
    vgg.summary()
    # Freeze the layers except the last 2 layers
    for layer in vgg.layers[:-5]:
        layer.trainable = False
    # Check the trainable status of the individual layers
    for layer in vgg.layers:
        print(layer, layer.trainable)
    # Create the model
    model = models.Sequential()
    # Add the vgg convolutional base model
    model.add(vgg)
    # Add new layers
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    male_filenames = [x for x in os.listdir(data_path + '/male/') if 'DS_Store' not in x]
    female_filenames = [x for x in os.listdir(data_path + '/female/') if 'DS_Store' not in x]
    train_males = np.array([cv2.imread(data_path + '/male/' + filename) for filename in male_filenames])
    train_females = np.array([cv2.imread(data_path + '/female/' + filename) for filename in female_filenames])
    train_males = np.array([cv2.resize(img, (218, 178), interpolation=cv2.INTER_AREA) for img in train_males])
    train_females = np.array([cv2.resize(img, (218, 178), interpolation=cv2.INTER_AREA) for img in train_females])
    X_train = np.concatenate([train_males, train_females])
    y_male = np.array([0 for _ in range(len(train_males))])
    y_female = np.array([1 for _ in range(len(train_females))])
    y_train = np.concatenate([y_male, y_female])
    train_data, train_labels = shuffle(X_train, y_train)

    history = model.fit(train_data, train_labels, batch_size=12, epochs=epochs, verbose=1, validation_split=0.2)
    with open('history_VGG.pkl', 'wb') as file:
        pickle.dump(history, file)
    model.save(model_path)
    print('model saved')


def train_dnn(data_path, model_path, epochs):
    IMG_HEIGHT = 178 // 4
    IMG_WIDTH = 218 // 4
    model = Sequential()
    model.add(Dense(1024, input_shape=(IMG_HEIGHT*IMG_WIDTH,), activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    male_filenames = [x for x in os.listdir(data_path + '/male/') if 'DS' not in x]
    female_filenames = [x for x in os.listdir(data_path + '/female/') if 'DS' not in x]
    train_males = np.array([cv2.imread(data_path + '/male/' + filename) for filename in male_filenames])
    train_females = np.array([cv2.imread(data_path + '/female/' + filename) for filename in female_filenames])
    train_males = np.array([cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (IMG_HEIGHT,IMG_WIDTH)).flatten() for img in train_males])
    train_females = np.array([cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (IMG_HEIGHT,IMG_WIDTH)).flatten() for img in train_females])
    X_train = np.concatenate([train_males, train_females])
    y_male = np.array([0 for _ in range(len(train_males))])
    y_female = np.array([1 for _ in range(len(train_females))])
    y_train = np.concatenate([y_male, y_female])
    train_data, train_labels = shuffle(X_train, y_train)
    history = model.fit(train_data, train_labels, batch_size=12, epochs=epochs, verbose=1, validation_split=0.2)
    with open('history_DNN.pkl', 'wb') as file:
        pickle.dump(history, file)
    model.save(model_path)
    print('model saved')


def preprocess_for_dnn(image):
    # img = preprocess_input(image)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # smaller = np.resize(cv2.resize(image, (50, 50)), (50, 50, 1))
    smaller = cv2.resize(image, (50, 50))
    return smaller.flatten()


def shuffle(X, y):
    indices = list(range(len(y)))
    np.random.shuffle(indices)
    train_data = X[indices]
    train_labels = y[indices]
    return train_data, train_labels


def train_cnn(data_path, model_path, epochs):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(178, 218, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    male_filenames = [x for x in os.listdir(data_path + '/male/') if 'DS_Store' not in x]
    female_filenames = [x for x in os.listdir(data_path + '/female/') if 'DS_Store' not in x]
    train_males = np.array([cv2.imread(data_path + '/male/' + filename) for filename in male_filenames])
    train_females = np.array([cv2.imread(data_path + '/female/' + filename) for filename in female_filenames])
    train_males = np.array([cv2.resize(img, (218, 178), interpolation=cv2.INTER_AREA) for img in train_males])
    train_females = np.array([cv2.resize(img, (218, 178), interpolation=cv2.INTER_AREA) for img in train_females])
    X_train = np.concatenate([train_males, train_females])
    y_male = np.array([0 for _ in range(len(train_males))])
    y_female = np.array([1 for _ in range(len(train_females))])
    y_train = np.concatenate([y_male, y_female])
    train_data, train_labels = shuffle(X_train, y_train)
    history = model.fit(train_data, train_labels, batch_size=12, epochs=epochs, verbose=1, validation_split=0.2)
    with open('history_CNN.pkl', 'wb') as file:
        pickle.dump(history, file)
    model.save(model_path)
    print('model saved')
