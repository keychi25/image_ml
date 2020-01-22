import tensorflow as tf
import tflearn

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from keras.models import load_model

from config import Config

def create_cnn_net():
    '''
    CNNを二層で作成する
    '''
    ## ニューラルネットワークの作成
    tf.reset_default_graph()  # 初期化
    net = input_data(shape=([None, 32, 32, 1]))
    # 中間層
    # 畳み込み層
    net = conv_2d(net, 32, 5, activation='relu')
    #プーリング層
    net = max_pool_2d(net, 2)
    #畳み込み層２
    net = conv_2d(net, 64 ,5 ,activation='relu')
    # プーリング層２
    net = max_pool_2d(net, 2)
    net = fully_connected(net ,128, activation='relu')
    net = dropout(net,0.5)
    # 出力層
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='sgd', learning_rate=0.5, loss='categorical_crossentropy')
    return net