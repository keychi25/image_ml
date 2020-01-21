from flask import render_template, request, redirect, url_for, send_from_directory, Blueprint
import os

from config import Config
from service.image_process import train, cnn_net, cnn_model
SAVE_DIR = Config.SAVE_DIR
# "root"という名前でBlueprintオブジェクトを生成
web = Blueprint('root', __name__, url_prefix='/')

@web.route('/')
def index():
    return render_template('index.html', images=os.listdir(SAVE_DIR)[::-1])

@web.route('/trains')
def trains():
    X_train, y_train = train()
    net = cnn_net()
    model = cnn_model(net, X_train, y_train)
    return redirect('/')


