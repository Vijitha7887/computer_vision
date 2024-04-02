"""
todo:
    done - fix y_proba, change output to be 1 neuron instead of 2
    beautify evaluation of classification: compare 0s and 1s instead of strings like 'male' and 'female'
    implement post-regression activation functions
    fix classification for dnn
    WARNING This does not produce image output. For that, see acomputervision/larger_face.py
    rename variables sensible - right now in the classify they are all called train stuff
    look into why roc for dnn is basically just two points - for some reason DNN always predicts the same number
"""

import os
import cv2
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from scipy.optimize import curve_fit


def get_rescaling_function(threshold=0.5, display=False):
    scale = 2 * threshold
    shift = 0
    if threshold > 0.5:
        scale = (1 - threshold) * 2
        shift = threshold * 2 - 1
    xcoords = np.array([0., 0.35, 0.5, 0.65, 1.]) * scale + shift
    ycoords = np.array([0., 0.2, 0.5, 0.8, 1.])

    def func(x, a, b, c, d):
        return a * x ** 3 + b * x ** 2 + c * x + d

    def tanh_func(x, a, b, c, d):
        return np.tanh(a * x + b) * c + d

    popt, pcov = curve_fit(tanh_func, xcoords, ycoords, p0=[1, 0, 1, 0])
    if display:
        xsample = np.linspace(0, 1, 1000)
        ysample = tanh_func(xsample, *popt)
        plt.figure(2)
        plt.plot(xsample, ysample)
        plt.plot(xcoords, ycoords, 'o')
        plt.title('Probability rescaling using tanh function')
        plt.xlabel('old probability values')
        plt.ylabel('new probability values')
        plt.savefig('gradation_function.png')

    return tanh_func, popt

def classification(model_path):
    post_activation = False

    # model = tf.keras.models.load_model(model_path)

    # data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
    # # generate data for test set of images
    # test_generator = data_generator.flow_from_directory(
    #     '/Users/kristofgalambos/Downloads/archive/test',
    #     target_size=(178, 218),
    #     batch_size=1,
    #     class_mode='binary',
    #     shuffle=False)
    # # obtain predicted activation values for the last dense layer
    # test_generator.reset()
    # y_proba = np.array(model.predict_generator(test_generator, verbose=1, steps=1000)).flatten()
    # # determine the maximum activation value for each sample
    # # y_proba = np.array([x[0] for x in pred])  # 1 for male and 0 for female
    # y_pred = np.array([round(x) for x in y_proba])
    # predicted_class_indices = y_pred
    # # label each predicted value to correct gender
    # labels = (test_generator.class_indices)
    # labels = dict((v, k) for k, v in labels.items())
    # predictions = [labels[k] for k in predicted_class_indices]
    # # format file names to simply male or female
    # filenames = test_generator.filenames
    # filenz = [0]
    # for i in range(0, len(filenames)):
    #     filenz.append(filenames[i].split('/')[0])
    # filenz = filenz[1:]

    IMG_HEIGHT = 218
    IMG_WIDTH = 178
    MAX_TEST_SIZE = 500
    model = tf.keras.models.load_model(model_path)

    data_path = '/Users/kristofgalambos/Downloads/archive/test'
    male_filenames = [x for x in os.listdir(data_path + '/male/') if 'DS' not in x]
    female_filenames = [x for x in os.listdir(data_path + '/female/') if 'DS' not in x]
    train_males = np.array([cv2.imread(data_path + '/male/' + filename) for filename in male_filenames])
    train_females = np.array([cv2.imread(data_path + '/female/' + filename) for filename in female_filenames])
    train_males = np.array(
        [cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH)) for img in train_males])[:MAX_TEST_SIZE]
    train_females = np.array(
        [cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH)) for img in train_females])[:MAX_TEST_SIZE]
    X_train = np.concatenate([train_males, train_females])
    y_male = np.array([0 for _ in range(len(train_males))])
    y_female = np.array([1 for _ in range(len(train_females))])
    y_train = np.concatenate([y_male, y_female])
    # if 'vgg' in model_path:
    #     y_train = y_train * (-1) + 1
    train_data, train_labels = shuffle(X_train, y_train)
    y_proba = model.predict(train_data).flatten()
    y_true = train_labels

    def threshold_optimizer(thresh, params):
        ytrue, yproba = params
        func, popt = get_rescaling_function(thresh)
        yproba = func(yproba, *popt)
        return accuracy_score(ytrue, yproba)

    if post_activation:
        p_opt, p_cov = curve_fit(threshold_optimizer, (y_true, y_proba), 1, p0=0.5)

        optimized_threshold = p_opt[0]
        print('optimized threshold =', optimized_threshold)
        rescaler, poptimal = get_rescaling_function(optimized_threshold)
        y_proba = rescaler(y_proba, *poptimal)
    y_pred = np.array([round(x) for x in y_proba])

    # shift = 0 # use different labellings for vgg
    # if 'cnn' not in model_path and 'dnn' not in model_path:
    #     shift = 1
    # y_true = np.array([1-shift if x == 'female' else 0+shift for x in filenz])
    print('accuracy_score:', accuracy_score(y_true, y_pred))
    print('auc score:', roc_auc_score(y_true, y_proba))
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    if 'cnn' in model_path:
        fname = 'roc_cnn.pkl'
    elif 'dnn' in model_path:
        fname = 'roc_dnn.pkl'
    else:
        fname = 'roc_vgg.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(fpr, f)
        pickle.dump(tpr, f)
        pickle.dump(thresholds, f)
    plt.figure(1)
    plt.title('ROC curve')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.plot(fpr, tpr)
    xref = np.linspace(0, 1, 10)
    yref = xref
    plt.plot(xref, yref, 'k--')
    plt.savefig('ROC_curve.png')

    # # determine the test set accuracy
    # match = []
    # for i in range(0, len(filenames)):
    #     match.append(filenz[i] == predictions[i])
    # print('accuracy:', match.count(True) / len(filenames))
    # results = pd.DataFrame({"Filename": filenz, "Predictions": predictions})
    # results.to_csv("GenderID_test_results.csv", index=False)


def classification_dnn(model_path):
    """just to account for the discrepancy that DNN expects 1D series of pixels"""
    IMG_HEIGHT = 178 // 4
    IMG_WIDTH = 218 // 4
    model = tf.keras.models.load_model(model_path)

    data_path = '/Users/kristofgalambos/Downloads/archive/test'
    male_filenames = [x for x in os.listdir(data_path + '/male/') if 'DS' not in x]
    female_filenames = [x for x in os.listdir(data_path + '/female/') if 'DS' not in x]
    train_males = np.array([cv2.imread(data_path + '/male/' + filename) for filename in male_filenames])
    train_females = np.array([cv2.imread(data_path + '/female/' + filename) for filename in female_filenames])
    train_males = np.array(
        [cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (IMG_HEIGHT, IMG_WIDTH)).flatten() for img in train_males])
    train_females = np.array(
        [cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (IMG_HEIGHT, IMG_WIDTH)).flatten() for img in train_females])
    X_train = np.concatenate([train_males, train_females])
    y_male = np.array([0 for _ in range(len(train_males))])
    y_female = np.array([1 for _ in range(len(train_females))])
    y_train = np.concatenate([y_male, y_female])
    train_data, train_labels = shuffle(X_train, y_train)

    print(train_data[0, 0])
    print(train_data[1, 0])
    print('train_data.shape=', train_data.shape)
    y_proba = model.predict(train_data).flatten()
    y_true = train_labels
    print(y_proba)

    y_pred = np.array([round(x) for x in y_proba])

    print('accuracy_score:', accuracy_score(y_true, y_pred))
    print(y_true.shape)
    print(y_pred.shape)
    print(y_pred)
    print('auc score:', roc_auc_score(y_true, y_proba))
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    fname = 'roc_dnn.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(fpr, f)
        pickle.dump(tpr, f)
        pickle.dump(thresholds, f)
    plt.figure(1)
    plt.title('ROC curve')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.plot(fpr, tpr)
    xref = np.linspace(0, 1, 10)
    yref = xref
    plt.plot(xref, yref, 'k--')
    plt.savefig('ROC_curve.png')


def shuffle(X, y):
    indices = list(range(len(y)))
    np.random.shuffle(indices)
    train_data = X[indices]
    train_labels = y[indices]
    return train_data, train_labels
