from digits_model.region_convultional import detect_digits_on_image
from digits_model.segmentation import selective_search


def detect_digits(original, is_negative=False):
    return detect_digits_on_image(original, is_negative)
