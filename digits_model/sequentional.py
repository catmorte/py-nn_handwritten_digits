import tensorflow as tf
import tensorflow.keras as krs
import pandas as pd

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