from PIL.ImageDraw import ImageDraw
from PIL import Image
import cv2
import tensorflow as tf


def selective_search(original):
    data = tf.keras.preprocessing.image.img_to_array(original)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(data)
    ss.switchToSelectiveSearchFast()
    # ss.switchToSelectiveSearchQuality()
    boxes = ss.process()
    draw = ImageDraw(original)
    print("found")
    for box in boxes:
        draw.rectangle(list(box), outline="#00f000", width=2)
    return original


if __name__ == "__main__":
    original = Image.open("./original.png")
    original = selective_search(original)
    original.save("./res.png")