import os
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
import tflearn
from PIL import Image

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from keras.models import load_model

from config import Config
TRAIN_DIR = Config.TRAIN_DIR

def canny(image):
    return cv2.Canny(image, 100, 200)

def train():
    train_dirs = ['pos','neg']

    X_train = []
    y_train = []

    for i, d in enumerate(train_dirs):
        files = os.listdir(TRAIN_DIR + '/' + d)
        for f in files:
            # 画像読み込み(グレースケール)
            gray_image = Image.open(TRAIN_DIR + '/' + d + '/' + f, 'r').convert('L')
            # 画像ファイルをピクセル値へ変換
            gray_image_px = np.array(gray_image)
            gray_image_flatten = gray_image_px.flatten().astype(np.float32)/255.0
            X_train.append(gray_image_flatten)

            # 正解データをone_hot形式へ変換
            tmp = np.zeros(2)
            tmp[i] = 1
            y_train.append(tmp)

    X_train = np.asarray(X_train)
    X_train = X_train.reshape([-1,32,32,1])
    y_train = np.asarray(y_train)
    return X_train, y_train

def cnn_net():
    ## ニューラルネットワークの作成
    tf.reset_default_graph()
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

def cnn_model(net, X_train, y_train):
    model = tflearn.DNN(net)
    model.fit(X_train, y_train, n_epoch=20, batch_size=100, validation_set=0.1, show_metric=True)
    model.save('my_model.h5')

    return model