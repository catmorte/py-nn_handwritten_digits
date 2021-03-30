# Handwritten digits NN
Dataset: https://www.kaggle.com/c/digit-recognizer/code
## Usage

```python
from digits_model.digits import predict_digit
import tensorflow as tf

...
tf.argmax(predict_digit("Path to image", color_mode="grayscale", is_negative=False), 1)
```

## Run re-learn
```
python ./digits_model/digits.py
```

## Run unittest
```
python -m unittest discover test
```
