import cv2
import numpy as np
from PIL import Image

def canny(image):
    '''
    エッジ検出する
    '''
    return cv2.Canny(image, 100, 200)

def to_array(req):
    '''
    array型に変換
    '''
    image_array = np.asarray(bytearray(req.read()), dtype=np.uint8)
    image = cv2.imdecode(image_array, 1)
    return image

def to_px(image):
    '''
    画像ファイルをピクセル値にする
    '''
    image_px = np.array(image)
    return image_px

def to_flatten(image_px): 
    '''
    ピクセル値を一次元にする
    '''
    image_flatten = image_px.flatten().astype(np.float32)/255.0
    return image_flatten

def to_gray(image):
    '''
    グレースケールにする
    '''
    image_gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    # gray_image = Image.fromarray(np.uint8(image_gray))
    return image_gray

def inversion(image):
    '''
    明暗を反転させる
    '''
    image = 255 - image
    return image