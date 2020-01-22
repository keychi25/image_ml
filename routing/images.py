from flask import render_template, request, redirect, url_for, send_from_directory, Blueprint, flash
import numpy as np
import cv2
from datetime import datetime
import os
import string
import random

from config import Config
from service.image_process import canny, to_gray, inversion, to_array

SAVE_DIR = Config.SAVE_DIR

# "root"という名前でBlueprintオブジェクトを生成
web = Blueprint('images', __name__, url_prefix='/images')

def random_str(n):
    return ''.join([random.choice(string.ascii_letters + string.digits) for i in range(n)])

@web.route('/<path:path>')
def send_js(path=None):
    return send_from_directory(SAVE_DIR, path)

@web.route('/upload/<path:detection>', methods=['POST'])
def upload(detection):
    try:
        if request.files['image']: # StrageFile
            stream = request.files['image'].stream
            img = to_array(stream)
            if detection == None:
                print('そのまま保存')
            if detection == 'edge':
                img = canny(img)
            elif detection == 'gray':
                img = to_gray(img)
            elif detection == 'inversion':
                img = to_gray(img)
                img = inversion(img)
            dt_now = datetime.now().strftime("%Y_%m_%d%_H_%M_%S_") + random_str(5)
            save_path = os.path.join(SAVE_DIR, dt_now + ".png")
            cv2.imwrite(save_path, img) 
    except IndexError as err:
        print('Bad Request:', err)
    except Exception as other:
        print('Something else Bloke:', other)
    return redirect(url_for('root.index'))
