import operator

import tensorflow as tf
from PIL import Image
import tensorflow.keras as krs
from PIL.ImageDraw import ImageDraw
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

from digits_model.digits import predict_digit_from_arrays


def sliding_window(image, step, ws):
    for y in range(0, image.shape[0] - ws[1], step):
        for x in range(0, image.shape[1] - ws[0], step):
            yield x, y, image[y:y + ws[1], x:x + ws[0]]


def image_pyramid(image, scale, min_size):
    yield image
    while True:
        w = int(image.shape[1] / scale)
        image = resize(image, width=w)
        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break
        yield image


def resize(image, width=None, height=None):
    if width is None and height is None:
        return image
    (h, w) = image.shape[:2]
    resW, resH = width, height
    if width is not None:
        r = width / float(w)
        resH = int(h * r)
    else:
        r = height / float(h)
        resW = int(w * r)

    resizing = tf.keras.layers.experimental.preprocessing.Resizing(height=resH, width=resW, interpolation='area')
    return resizing(image)


def non_max_suppression(boxes, predictions, labels, overlap_thresh=0.3):
    if len(boxes) == 0:
        return [], []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if predictions is not None:
        idxs = predictions

    # sort the indexes
    idxs = np.argsort(idxs)
    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

    return boxes[pick].astype("int"), labels[pick]


if __name__ == "__main__":
    original_resized_width = 300
    model_input_size = (28, 28)
    window_step = 12
    pyramid_scale = 1.5
    min_pyramid_size = (28, 28)
    window_size = (28, 28)
    prediction_level = 0.90
    with Image.open("./original_2.png") as original:
        global_scale = original.width / original_resized_width
        original_resized = original.convert('L')
        original_resized = tf.keras.preprocessing.image.img_to_array(original_resized)
        original_resized = resize(original_resized, width=original_resized_width)
        (H, W) = original_resized.shape[:2]
        regions = []
        locations = []

        for image in image_pyramid(original_resized, scale=pyramid_scale, min_size=min_pyramid_size):
            scale = W / float(image.shape[1])
            for (x, y, roiOrig) in sliding_window(image, step=window_step, ws=window_size):
                x = int(x * scale)
                y = int(y * scale)
                w = int(window_size[0] * scale)
                h = int(window_size[1] * scale)
                roi = resize(roiOrig, *model_input_size)
                regions.append(roi)
                locations.append((x, y, x + w, y + h))

        predictions = predict_digit_from_arrays(regions, is_negative=False)
        boxes_and_predictions_per_label = {}

        for (i, p) in enumerate(predictions):
            label, prob = max(enumerate(p), key=operator.itemgetter(1))
            if prob >= prediction_level:
                box = locations[i]
                L = boxes_and_predictions_per_label.get(label, [])
                L.append((box, prob))
                boxes_and_predictions_per_label[label] = L

        boxes = np.array([p[0] for key in boxes_and_predictions_per_label for p in boxes_and_predictions_per_label[key]])
        all_predictions = np.array([p[1] for key in boxes_and_predictions_per_label for p in boxes_and_predictions_per_label[key]])
        names = np.array([key for key in boxes_and_predictions_per_label for p in boxes_and_predictions_per_label[key]])
        draw = ImageDraw(original)

        boxes, names = non_max_suppression(boxes, all_predictions, names, overlap_thresh=0.1)
        for j in range(len(boxes)):
            box = boxes[j]
            draw.ellipse(list(box * global_scale), outline="#00f000", width=2)
            draw.text(
                [(box[0] + box[2]) * global_scale / 2, box[1] * global_scale],
                f"{names[j]}",
                fill="#00f000",
                stroke_width=5)
        original.save(f"./draw.png")
