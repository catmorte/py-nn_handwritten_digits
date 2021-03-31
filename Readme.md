# Handwritten digits NN
Dataset: https://www.kaggle.com/c/digit-recognizer/code

Deployed: https://sleepy-springs-74218.herokuapp.com/
#### Usage

```python
from digits_model.digits import predict_digit_from_file
import tensorflow as tf

...
tf.argmax(predict_digit_from_file("Path to image", color_mode="grayscale", is_negative=False), 1)
```

```python
from digits_model.digits import predict_digit_from_img
import tensorflow as tf
from PIL import Image
...
tf.argmax(predict_digit_from_img(Image.open('...'), is_negative=False), 1)
```

#### Run re-learn
```
python ./digits_model/digits.py
```

## Docker
#### build
```
docker build -t digits_nn_sandbox .
```
#### run
```
docker run -d -p 8080:8080 digits_nn_sandbox
```

## Heroku
#### Build
```
heroku container:push web
```
#### Deploy
```
heroku container:release web
```
#### Logs
```
heroku logs --tail
```