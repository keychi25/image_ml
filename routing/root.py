from flask import render_template, request, redirect, url_for, send_from_directory, Blueprint
import numpy as np
import cv2
from datetime import datetime
import os
import string
import random

from config import Config
from service.image_process import canny, train, cnn_net, cnn_model

SAVE_DIR = Config.SAVE_DIR
# "root"という名前でBlueprintオブジェクトを生成
web = Blueprint('root', __name__, url_prefix='/')

def random_str(n):
    return ''.join([random.choice(string.ascii_letters + string.digits) for i in range(n)])

@web.route('/')
def index():
    return render_template('index.html', images=os.listdir(SAVE_DIR)[::-1])

@web.route('/trains')
def trains():
    X_train, y_train = train()
    net = cnn_net()
    model = cnn_model(net, X_train, y_train)
    
    return redirect('/')

@web.route('/images/<path:path>')
def send_js(path):
    return send_from_directory(SAVE_DIR, path)

@web.route('/upload', methods=['POST'])
def upload():
    if request.files['image']:
        stream = request.files['image'].stream
        img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, 1)
        img = canny(img)
        dt_now = datetime.now().strftime("%Y_%m_%d%_H_%M_%S_") + random_str(5)
        save_path = os.path.join(SAVE_DIR, dt_now + ".png")
        cv2.imwrite(save_path, img)
        print("save", save_path)

        return redirect('/')

