import tensorflow as tf
import tensorflow.keras as krs
import pandas as pd
import numpy as np

if __name__ == "__main__":
    def dataframe_to_dataset(df):
        df = df.copy()
        labels = df.pop("label")
        ds = tf.data.Dataset.from_tensor_slices((tf.cast(df.values, tf.int32), labels))
        return ds


    dataframe = pd.read_csv('./assets/train.csv')
    val_dataframe = dataframe.sample(frac=0, random_state=1337)
    train_dataframe = dataframe.drop(val_dataframe.index)
    train_ds = dataframe_to_dataset(train_dataframe)
    val_ds = dataframe_to_dataset(val_dataframe)
    train_ds = train_ds.batch(32)
    val_ds = val_ds.batch(32)

    model = krs.Sequential()
    model.add(krs.layers.Dense(392, activation=tf.nn.relu))
    model.add(krs.layers.Dense(196, activation=tf.nn.relu))
    model.add(krs.layers.Dense(98, activation=tf.nn.relu))
    model.add(krs.layers.Dense(10, activation=tf.nn.softmax))
    model.compile(
        optimizer=krs.optimizers.Adam(),
        loss=krs.losses.SparseCategoricalCrossentropy(),
        metrics=[krs.metrics.SparseCategoricalAccuracy()],
    )

    model.fit(
        train_ds.repeat(),
        epochs=16,
        steps_per_epoch=1024,
    )

    model.save('./assets/trained_model')
else:
    model = krs.models.load_model('./assets/trained_model')


def predict_digit_from_file(image_path, *, color_mode="grayscale", is_negative=False):
    img = krs.preprocessing.image.load_img(image_path, color_mode=color_mode, target_size=(28, 28))
    data = np.asarray(img, dtype="int32")
    if not is_negative:
        data = [255 - x for x in data]
    data = np.asarray(data).reshape((1, 784))
    return model.predict(data)[0]


def predict_digit_from_img(img, *, is_negative=False):
    img = img.resize(size=(28, 28)).convert('L')
    data = tf.keras.preprocessing.image.img_to_array(img)
    if not is_negative:
        data = [255 - x for x in data]
    data = np.asarray(data).reshape((1, 784))
    return model.predict(data)[0]
