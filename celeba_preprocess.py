
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess(play=True):
    if play:
        SAMPLE_NUMBER = 1000
        SUFFIX = '_play'
    else:
        SAMPLE_NUMBER = 10000
        SUFFIX = ''
    print('preprocessing')
    basepath = '/Users/kristofgalambos/Downloads/archive/'
    df = pd.read_csv(basepath + 'list_attr_celeba.csv')
    filenames = df[:SAMPLE_NUMBER]['image_id']
    labels = np.array(df['Male'][:SAMPLE_NUMBER] == -1).astype(int)
    print(labels.shape)
    print(labels[int(SAMPLE_NUMBER/2)])

    X_train, X_test, y_train, y_test = train_test_split(filenames, labels, test_size=0.2)
    for filename, label in zip(X_train, y_train):
        if label == 0:
            shutil.copyfile(basepath + 'img_align_celeba/img_align_celeba/' + filename, basepath + 'train' + SUFFIX + '/male/' + filename)
        else:
            shutil.copyfile(basepath + 'img_align_celeba/img_align_celeba/' + filename, basepath + 'train' + SUFFIX + '/female/' + filename)
    for filename, label in zip(X_test, y_test):
        if label == 0:
            shutil.copyfile(basepath + 'img_align_celeba/img_align_celeba/' + filename, basepath + 'test' + SUFFIX + '/male/' + filename)
        else:
            shutil.copyfile(basepath + 'img_align_celeba/img_align_celeba/' + filename, basepath + 'test' + SUFFIX + '/female/' + filename)
    print('Preprocessing done')
