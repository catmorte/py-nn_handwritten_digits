import tensorflow as tf
import tensorflow.keras as krs
import numpy as np

model = krs.models.load_model('./assets/trained_model')
modelDe = krs.models.load_model('./assets/trained_model_dec')


def predict_digit_from_file(image_path, *, color_mode="grayscale", is_negative=False):
    img = krs.preprocessing.image.load_img(image_path, color_mode=color_mode, target_size=(28, 28))
    data = np.asarray(img, dtype="int32")
    if not is_negative:
        data = [255 - x for x in data]
    data = np.asarray(data).reshape((1, 784))
    return model.predict(data)[0]


def predict_digit_from_img(img, *, is_negative=False):
    img = img.resize(size=(28, 28)).convert('L')
    data = tf.keras.preprocessing.image.img_to_array(img)
    if not is_negative:
        data = [255 - x for x in data]
    data = np.asarray(data).reshape((1, 784))
    return model.predict(data)[0]


def predict_digit_from_dig_dec(digit):
    data = np.zeros((1, 10))
    data[0, digit] = 1
    return modelDe.predict(data)[0].reshape(28, 28, 1)
