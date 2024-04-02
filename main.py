"""
todo:
    mostly done - implement primitive DNN and CNN
    done - upload to github, create gitignore
    run once every model with many epochs, long processing for final results
"""

from celeba_preprocess import preprocess
from train_model import train_vgg, train_cnn, train_dnn
from classify import classification, classification_dnn
import time

start_time = time.perf_counter()

PREPROCESS = False
TRAIN_MODEL = False
CLASSIFY = True
MODEL_TYPE = 'vgg'
# MODEL_TYPE = 'cnn'
# MODEL_TYPE = 'dnn'

if PREPROCESS:
    preprocess(play=False)

if TRAIN_MODEL:
    if MODEL_TYPE == 'cnn':
        train_cnn(data_path='/Users/kristofgalambos/Downloads/archive/train',
                  model_path='models/mac_model_celeba_cnn_10', epochs=10)
    elif MODEL_TYPE == 'dnn':
        train_dnn(data_path='/Users/kristofgalambos/Downloads/archive/train',
                  model_path='models/mac_model_celeba_dnn_10', epochs=10)
    else:
        train_vgg(data_path='/Users/kristofgalambos/Downloads/archive/train',
                  model_path='models/mac_model_celeba_vgg_10', epochs=10)

if CLASSIFY:
    if MODEL_TYPE == 'cnn':
        classification('models/mac_model_celeba_cnn_10')
    elif MODEL_TYPE == 'dnn':
        classification_dnn('models/mac_model_celeba_dnn_10')
    else:
        classification('models/mac_model_celeba_vgg_10')

end_time = time.perf_counter()
print('program finished in', end_time - start_time, 'seconds')
print('PROGRAM ENDS HERE')
