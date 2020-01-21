import os

from flask import Flask, Blueprint, render_template
from routing import root, images
from flask_cors import CORS
from config import Config

if not os.path.isdir(Config.SAVE_DIR):
    os.makedirs(Config.SAVE_DIR)

app = Flask(__name__, static_url_path="")
CORS(app)
# 分割先のルーティング(Blueprint)を登録
app.register_blueprint(root.web) #/
app.register_blueprint(images.web) #/

@app.errorhandler(404)
def page_not_found(error):
    return render_template('page_not_found.html'), 404

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8888)
