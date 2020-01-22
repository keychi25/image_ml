import tensorflow.compat.v1 as tf
import tflearn

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from keras.models import load_model

from config import Config
TRAIN_DIR = Config.TRAIN_DIR

def cnn_model(net, X_train, y_train):
    model = tflearn.DNN(net)
    model.fit(X_train, y_train, n_epoch=20, batch_size=100, validation_set=0.1, show_metric=True)
    model.save('my_model.h5')
    return model