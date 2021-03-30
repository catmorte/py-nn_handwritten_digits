import unittest
from digits_model.digits import predict_digit
import tensorflow as tf


class TestModel(unittest.TestCase):
    def test_images(self):
        self.assertEqual(tf.argmax(predict_digit(f'./assets/validation_images/0.png', color_mode="grayscale", is_negative=False), 1), 0)
        self.assertEqual(tf.argmax(predict_digit(f'./assets/validation_images/1.png', color_mode="grayscale", is_negative=False), 1), 1)
        self.assertEqual(tf.argmax(predict_digit(f'./assets/validation_images/2.png', color_mode="grayscale", is_negative=False), 1), 2)
        self.assertEqual(tf.argmax(predict_digit(f'./assets/validation_images/3.png', color_mode="grayscale", is_negative=False), 1), 3)
        self.assertEqual(tf.argmax(predict_digit(f'./assets/validation_images/4.png', color_mode="grayscale", is_negative=False), 1), 4)
        self.assertEqual(tf.argmax(predict_digit(f'./assets/validation_images/5.png', color_mode="grayscale", is_negative=False), 1), 5)
        self.assertEqual(tf.argmax(predict_digit(f'./assets/validation_images/6.png', color_mode="grayscale", is_negative=False), 1), 6)
        self.assertEqual(tf.argmax(predict_digit(f'./assets/validation_images/8.png', color_mode="grayscale", is_negative=False), 1), 8)
        self.assertEqual(tf.argmax(predict_digit(f'./assets/validation_images/9.png', color_mode="grayscale", is_negative=False), 1), 9)
        self.assertEqual(tf.argmax(predict_digit(f'./assets/validation_images/7.png', color_mode="grayscale", is_negative=False), 1), 7)


if __name__ == '__main__':
    unittest.main()
