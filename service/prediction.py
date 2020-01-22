import keras

def cnn_prediction(image):
    model = keras.models.load_model('model.h5', compile=False)
    predictions = model.predict(image)
    print(predictions)