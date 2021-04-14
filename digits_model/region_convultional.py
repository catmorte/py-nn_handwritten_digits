import tensorflow as tf
from PIL import Image
import tensorflow.keras as krs
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

from digits_model.digits import predict_digit_from_images


def sliding_window(image, step, ws):
    for y in range(0, image.shape[0] - ws[1], step):
        for x in range(0, image.shape[1] - ws[0], step):
            yield x, y, image[y:y + ws[1], x:x + ws[0]]


def image_pyramid(image, scale=1.5, min_size=(28, 28)):
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


def non_max_suppression(boxes, probs=None, overlap_thresh=0.3):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
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
    if probs is not None:
        idxs = probs

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

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_thresh)[0])))

    # return only the bounding boxes that were picked
    return boxes[pick].astype("int")


WIDTH = 600
PYR_SCALE = 1.5
WIN_STEP = 16
ROI_SIZE = (25, 25)
INPUT_SIZE = (28, 28)
if __name__ == "__main__":
    orig = Image.open("./original.png").convert('L')
    orig = tf.keras.preprocessing.image.img_to_array(orig)
    orig = resize(orig, width=WIDTH)
    tf.keras.preprocessing.image.array_to_img(orig).save("original_resized.png")

    (H, W) = orig.shape[:2]
    pyramid = image_pyramid(orig, scale=PYR_SCALE, min_size=ROI_SIZE)
    rois = []
    locs = []

    pyr_i = 0
    for image in pyramid:
        pyr_i += 1
        scale = W / float(image.shape[1])
        roi_i = 0
        for (x, y, roiOrig) in sliding_window(image, WIN_STEP, ROI_SIZE):
            roi_i += 1
            x = int(x * scale)
            y = int(y * scale)
            w = int(ROI_SIZE[0] * scale)
            h = int(ROI_SIZE[1] * scale)
            roi = resize(roiOrig, *INPUT_SIZE)
            rois.append(roi)
            locs.append((x, y, x + w, y + h))

    # rois = np.array(rois, dtype="float32")

    preds = predict_digit_from_images(rois, is_negative=False)
    labels = {}
    for (i, p) in enumerate(preds):
        # grab the prediction information for the current ROI
        (imagenetID, label, prob) = p[0]
        # filter out weak detections by ensuring the predicted probability
        # is greater than the minimum probability
        if prob >= 0.95:
            # grab the bounding box associated with the prediction and
            # convert the coordinates
            box = locs[i]
            # grab the list of predictions for the label and add the
            # bounding box and probability to the list
            L = labels.get(label, [])
            L.append((box, prob))
            labels[label] = L

    boxes = np.array([p[0] for p in labels[label]])
    proba = np.array([p[1] for p in labels[label]])
    boxes = non_max_suppression(boxes, proba)

    # loop over all bounding boxes that were kept after applying
    # non-maxima suppression
    # for (startX, startY, endX, endY) in boxes:
    #     # draw the bounding box and label on the image
    #     cv2.rectangle(clone, (startX, startY), (endX, endY),
    #                   (0, 255, 0), 2)
    #     y = startY - 10 if startY - 10 > 10 else startY + 10
    #     cv2.putText(clone, label, (startX, y),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    # # show the output after apply non-maxima suppression
    # cv2.imshow("After", clone)
    # cv2.waitKey(0)
