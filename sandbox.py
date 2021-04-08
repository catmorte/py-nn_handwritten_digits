from io import BytesIO

import numpy
from PIL import Image
from PIL.ImageOps import invert
from flask import Flask, jsonify, render_template, request, send_file
from digits_model.digits import predict_digit_from_img, predict_digit_from_dig_dec, predict_bi_digit_from_img
import tensorflow as tf
import os
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recognize', methods=['POST'])
def recognize():
    img = Image.open(request.files['file'].stream)
    prediction = predict_digit_from_img(img, is_negative=False)
    bi_prediction = predict_bi_digit_from_img(img, is_negative=False)
    tf.keras.preprocessing.image.array_to_img(bi_prediction).save("./bi_img.png")
    return str([x for x in prediction])


@app.route('/draw/<int:digit>/', methods=['GET'])
def recognize_de(digit):
    prediction = predict_digit_from_dig_dec(digit)
    img_io = BytesIO()
    img = invert(tf.keras.preprocessing.image.array_to_img(prediction)).convert('1')
    img.save(img_io, 'PNG', quality=100)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')


if __name__ == "__main__":
    app.run('0.0.0.0', port=int(os.environ.get('PORT', 8081)))
