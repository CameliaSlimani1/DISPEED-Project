import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from src.python.utils.reports_generations import *


def build_and_train_rf(x_train, y_train, structure)  :
    forest = RandomForestClassifier(n_estimators=structure["n_estimators"], max_depth=structure["max_depth"])
    forest.fit(x_train, y_train)
    return forest

def build_and_train_nn(x_train, y_train, x_test, y_test, structure, metrics, loss: str, batch_size=32, epochs=20)  :
    model = tf.keras.Sequential()
    for layer in structure['layers']:
        layer_type = layer['type']
        layer_params = layer['params']
        if layer_type == 'dense':
            model.add(tf.keras.layers.Dense(**layer_params))
        elif layer_type == 'conv2d':
            model.add(tf.keras.layers.Conv2D(**layer_params))
        elif layer_type == 'conv1d':
            model.add(tf.keras.layers.Conv1D(**layer_params))
        elif layer_type == 'maxpool1d':
            model.add(tf.keras.layers.MaxPool1D(**layer_params))
        elif layer_type == 'maxpool2d':
            model.add(tf.keras.layers.MaxPool2D(**layer_params))
        elif layer_type == 'dropout':
            model.add(tf.keras.layers.Dropout(**layer_params))
        elif layer_type == "flatten" :
            model.add(tf.keras.layers.Flatten())
        else:
            print("Unsupported layer")
    model.compile(loss=loss, optimizer=tf.optimizers.Adam(), metrics=metrics)
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
    val_loss = history.history['val_loss']
    train_loss = history.history['loss']
    epochs = range(1,epochs+1, 1)
    print(epochs)
    print(val_loss)
    print(train_loss)
    plotLosses(train_loss, val_loss, epochs, "epoch", "Loss", "b", "r", 0)
    return model
