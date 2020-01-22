import os
import numpy as np
from PIL import Image

from config import Config

TRAIN_DIR = Config.TRAIN_DIR

def images_train():
    '''
    画像をトレーニングする
    '''
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