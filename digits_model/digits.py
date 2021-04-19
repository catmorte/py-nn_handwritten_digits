import tensorflow as tf
import tensorflow.keras as krs
import numpy as np

model = krs.models.load_model('./assets/trained_model_v2')
modelDe = krs.models.load_model('./assets/trained_model_dec_v2')


def predict_digit_from_img(img, *, is_negative=False):
    img = img.resize(size=(28, 28)).convert('L')
    data = tf.keras.preprocessing.image.img_to_array(img)
    if not is_negative:
        data = [255 - x for x in data]
    data = np.asarray(data).reshape((1, 784))
    return model.predict(data)[0]


def predict_digit_from_arrays(imgs, *, is_negative=False):
    data = [(np.asarray([255 - x if not is_negative else x for x in data]).reshape((1, 784))) for data in imgs]
    return model.predict(np.vstack(data))


def predict_digit_from_int(digit):
    data = np.zeros((1, 10))
    data[0, digit] = 1
    return modelDe.predict(data)[0].reshape(28, 28, 1)
