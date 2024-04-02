from keras import Sequential
from keras.layers import Conv2D, Dense


def prepare_data():
    pass

def get_model_cnn():
    model = Sequential()
    model.add(Conv2D(64))
    model.add(Conv2D(64))
    model.add(Dense(1, activation='sigmoid'))
    return model

def get_model_dnn():
    model = Sequential()
    model.add(Dense(128))
    model.add(Dense(128))
    return model
