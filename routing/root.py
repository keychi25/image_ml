from flask import render_template, request, redirect, url_for, send_from_directory, Blueprint
import os 
import numpy as np
from PIL import Image

from config import Config

from service.train import images_train
from service.create_net import create_cnn_net
from service.save_model import cnn_model
from service.prediction import cnn_prediction
from service.image_process import to_array

SAVE_DIR = Config.SAVE_DIR
# "root"という名前でBlueprintオブジェクトを生成
web = Blueprint('root', __name__, url_prefix='/')

@web.route('/')
def index(**kwargs):
    if 'content' in kwargs and 'category' in kwargs:
        flash(content, category=category)
    return render_template('index.html', images=os.listdir(SAVE_DIR)[::-1])

@web.route('/cnn/trains')
def trains():
    X_train, y_train = images_train()
    net = create_cnn_net()
    model = cnn_model(net, X_train, y_train)
    return redirect(url_for('root.index'))

@web.route('/cnn/predict/<path:img>')
def predict(img):
    test_path = os.path.join(SAVE_DIR, img)
    image = Image.open(test_path, 'r')
    # image = to_array(images_train)
    image = image.resize((32,32))
    image_px = np.array(image)
    image_px = image_px.flatten().astype(np.float32)/255.0
    image_px = image_px.reshape([-1, 32, 32, 1])

    X_train, y_train = images_train()
    net = create_cnn_net()
    model = cnn_model(net, X_train, y_train) 
    predictions = model.predict(image_px)
    print(predictions)
    # cnn_prediction(image_px)
    return redirect(url_for('root.index'))