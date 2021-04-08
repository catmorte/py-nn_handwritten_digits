import tensorflow as tf
import tensorflow.keras as krs
import pandas as pd

if __name__ == "__main__":
    def transform_labels_value_to_array(val):
        return [1 if val == i else 0 for i in range(10)]


    def dataframe_to_dataset(df):
        df = df.copy()
        labels = df.pop("label")
        labels = labels.transform(transform_labels_value_to_array)
        labels = pd.DataFrame(labels.to_list(), columns=[f"{i}" for i in range(10)])
        print(labels)
        ds = tf.data.Dataset.from_tensor_slices((labels.values, df))

        return ds


    dataframe = pd.read_csv('./assets/train.csv')
    val_dataframe = dataframe.sample(frac=0, random_state=1337)
    train_dataframe = dataframe.drop(val_dataframe.index)
    train_ds = dataframe_to_dataset(train_dataframe)
    val_ds = dataframe_to_dataset(val_dataframe)
    train_ds = train_ds.batch(1)
    val_ds = val_ds.batch(1)
    model = krs.Sequential()

    model.add(krs.layers.Dense(10, activation=tf.nn.relu, input_shape=(10,)))
    model.add(krs.layers.Dense(128, activation=tf.nn.relu))
    model.add(krs.layers.Dense(64, activation=tf.nn.relu))
    model.summary()
    model.add(krs.layers.Reshape((8, 8, 1)))
    model.add(tf.keras.layers.UpSampling2D((1, 1)))
    model.add(tf.keras.layers.Conv2D(2, (2, 2), activation=tf.nn.relu))
    model.add(tf.keras.layers.UpSampling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(2, (2, 2), activation=tf.nn.relu, kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.UpSampling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(1, (2, 2), activation=tf.nn.sigmoid, padding='same'))
    model.add(krs.layers.Flatten())

    model.summary()

    model.compile(
        optimizer=krs.optimizers.Adam(),
        loss=krs.losses.Huber(),
        metrics=[krs.metrics.BinaryCrossentropy()],
    )

    model.fit(
        train_ds.repeat(),
        epochs=32,
        steps_per_epoch=1024,
        validation_data=val_ds
    )
    model.save('./assets/trained_model_dec')
