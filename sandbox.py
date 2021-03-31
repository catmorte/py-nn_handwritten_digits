import numpy
from PIL import Image
from flask import Flask, jsonify, render_template, request
from digits_model.digits import predict_digit_from_img
import tensorflow as tf

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recognize', methods=['POST'])
def recognize():
    img = Image.open(request.files['file'].stream)
    prediction = predict_digit_from_img(img, is_negative=False)
    return str([x for x in prediction])


if __name__ == "__main__":
    app.run('0.0.0.0', port=8080)
