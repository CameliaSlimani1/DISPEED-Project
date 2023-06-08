import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, save_model
from src.utils.models_building import *
from sklearn.metrics import accuracy_score, f1_score
from hummingbird.ml import convert, load

class IDSModel():

    def __init__(self, name, type, structure):
        self.name = name
        self.type = type
        self.structure = structure
        self.model = None

    # Creates a new model
    def create_model(self, x_train, y_train, x_test, y_test):
        if self.type == 'RF' :
            self.model = build_and_train_rf(x_train, y_train, self.structure)
        elif self.type == 'DNN' or self.type == 'CNN':
            self.model = build_and_train_nn(x_train, y_train, x_test, y_test, self.structure, ['categorical_accuracy'], 'categorical_crossentropy', 32, 20)

    # Loads an already serialized model
    def load_ids_model(self, filepath):
        if self.type == 'RF':
            self.model = joblib.load(filepath)
        elif self.type == 'CNN' or self.type == 'DNN':
            self.model = load_model(filepath)

    # Serializes a model
    def save_ids_model(self, ):
        if self.type == 'RF':
            joblib.dump(self.model, f"../output/models/{self.type}/{self.name}")
        elif self.type == 'CNN' or self.type == 'DNN':
            self.model.save(f"../output/models/{self.type}/{self.name}.h5")




    # Performs inference on the test ensemble and returns accuracy and f1-score

    def get_security_metrics(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        if self.type =='CNN' or self.type=='DNN':
            y_pred= np.argmax(y_pred, axis=1)
            y_test= np.argmax(y_test, axis=1)
        acc = accuracy_score(y_pred, y_test)
        f1score = f1_score(y_pred, y_test, average='macro')
        return acc, f1score


    # Generates tflite model
    def generate_tflite_model(self, x_test, opt=False):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        if opt:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            def representative_dataset_generator():
                x = x_test[1:]
                for value in x:
                    yield [np.array(value, dtype=np.float32, ndmin=2)]

            converter.representative_dataset = representative_dataset_generator
        tflite_model = converter.convert()
        with open(f"../output/models/{self.type}/{self.name}.tflite", "wb") as f:
            f.write(tflite_model)


    def generate_rf_for_gpu(self):
        model = convert(self.model, 'pytorch')
        model.to('cuda')
        model.save(f"../output/models/RF/{self.name}_gpu")


    #TODO : generate RF for CPU






