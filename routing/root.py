from flask import render_template, request, redirect, url_for, send_from_directory, Blueprint
import os

from config import Config

from service.train import images_train
from service.create_net import create_cnn_net
from service.save_model import cnn_model
SAVE_DIR = Config.SAVE_DIR
# "root"という名前でBlueprintオブジェクトを生成
web = Blueprint('root', __name__, url_prefix='/')

@web.route('/')
def index(**kwargs):
    if 'content' in kwargs and 'category' in kwargs:
        flash(content, category=category)
    return render_template('index.html', images=os.listdir(SAVE_DIR)[::-1])

@web.route('/trains')
def trains():
    X_train, y_train = images_train()
    net = create_cnn_net()
    model = cnn_model(net, X_train, y_train)
    return redirect(url_for('root.index'))
