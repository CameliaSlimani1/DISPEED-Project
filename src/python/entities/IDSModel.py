import os.path

import joblib
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, save_model
from src.python.utils.models_building import *
from src.python.utils.optimize import *
from sklearn.metrics import accuracy_score, f1_score
from hummingbird.ml import convert, load
from sklearn.model_selection import ParameterGrid
import keras_tuner

class IDSModel():

    def __init__(self, name, type, structure):
        self.name = name
        self.type = type
        self.structure = structure
        self.model = None

    # Explore RF Space

    def explore_rf_models (self, x_train, y_train, x_test, y_test, x_val, y_val):
        params = { 'n_estimators': [5, 10, 25, 50, 75, 100],
                       'max_depth': [5, 10, 25 , 50, 75, 100]}

        param_grid = ParameterGrid(params)
        acc = 0
        best_acc = 0
        exploration_results = {"n_estimators":[], "max_depth":[],  "accuracy":[], "size":[]}
        for config in param_grid:
            print(config)
            rf = RandomForestClassifier(n_estimators=config['n_estimators'], max_depth=config['max_depth'])
            rf.fit(x_train, y_train)
            y_pred = rf.predict(x_test)
            acc = accuracy_score(y_test, y_pred)

            #calculate rf size by serializing it
            joblib.dump(rf, "tmp_rf.joblib")
            size = np.round(os.path.getsize('tmp_rf.joblib')/1024 / 1024, 2)
            os.remove('tmp_rf.joblib')

            print("config %s, accuracy : %f, size %f" %(config, acc, size))
            exploration_results["n_estimators"].append(config['n_estimators'])
            exploration_results["max_depth"].append(config['max_depth'])
            exploration_results["accuracy"].append(acc)
            exploration_results["size"].append(size)
            print(exploration_results)
            if acc > best_acc :
                best_acc = best_acc
                best_config = config
        #plot3D(exploration_results["n_estimators"],exploration_results["max_depth"], exploration_results["accuracy"], np.array(exploration_results["size"]))
        pareto, count =  findParetoFront(exploration_results)
        with open("../../output/explorations/%s_explorations.json"%(self.name), "w") as file:
            json.dump(exploration_results, file)
        with open(f"../../output/explorations/%s_pareto.json"%(self.name), "w") as file:
            json.dump(pareto, file)
        plotPareto(exploration_results["accuracy"], np.array(exploration_results["size"]), "Accuracy", "Size (MB)",
                   "bo")
        print(pareto)
        print(count)
        plotPareto_2(exploration_results["accuracy"],  pareto["accuracy"], np.array(exploration_results["size"]), np.array(pareto["size"]),"Accuracy", "Size (MB)", "bo", "ro", "Solutions", "Pareto Solutions")


        print(pareto)
        print(count)
        print(best_config)





    # DNN exploration



    def explore_dnn_models (self, x_train, y_train, x_test, y_test, x_val, y_val):
        def generate_dnn_model(hp):
            model = tf.keras.Sequential()
            for i in range(hp.Int("num_fc", 5, 8)):
                model.add(tf.keras.layers.Dense(
                    # Tune number of units separately.
                    units=hp.Int(f"units_{i}", min_value=256, max_value=1024, step=64),
                    kernel_initializer='glorot_uniform',
                    activation=hp.Choice(f"activation_{i}", ["relu"]),
                ))

            model.add(tf.keras.layers.Dense(10, activation='softmax', name='output'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            return model
        hpmodel = generate_dnn_model(keras_tuner.HyperParameters())
        dnn_tuner=keras_tuner.BayesianOptimization(generate_dnn_model,
            objective="accuracy",
            max_trials=50,
            executions_per_trial=1,
            directory="../tmp",
            project_name="dispeed"
        )
        dnn_tuner.search_space_summary()
        dnn_tuner.search(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=10, verbose=1)
        models = dnn_tuner.get_best_models(50)
        exploration_results = {"model_summary": [], "accuracy": [], "size" : [] }
        for i in range (50):
            model = models[i]
            y_predict = np.argmax(model.predict(x_test))
            acc = accuracy_score(y_test, y_predict)
            size = model.count_params()
            exploration_results["model_summary"].append(model.summary())
            exploration_results["accuracy"].append(acc)
            exploration_results["size"].append(size)

        plotPareto(exploration_results["accuracy"], np.array(exploration_results["size"]), "Accuracy", "Size (MB)",
                   "bo")
        pareto, count = findParetoFront(exploration_results)
        print(pareto)
        print(count)
        plotPareto_2(exploration_results["accuracy"], pareto["accuracy"], np.array(exploration_results["size"]),
                     np.array(pareto["size"]), "Accuracy", "Size (MB)", "bo", "ro", "Solutions", "Pareto Solutions")

        return exploration_results


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
            joblib.dump(self.model, f"../../output/models/{self.type}/{self.name}")
        elif self.type == 'CNN' or self.type == 'DNN':
            self.model.save(f"../../output/models/{self.type}/{self.name}.h5")




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
        with open(f"../../output/models/{self.type}/{self.name}.tflite", "wb") as f:
            f.write(tflite_model)


    def generate_rf_for_gpu(self):
        model = convert(self.model, 'pytorch')
        model.to('cuda')
        model.save(f"../../output/models/RF/{self.name}_gpu")


    #TODO : generate RF for CPU






