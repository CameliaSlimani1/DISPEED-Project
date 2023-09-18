from entities.Platform import Platforme
from entities.Implementation import Implementation
from entities.IDSModel import IDSModel
from entities.Dataset import Dataset
from entities.AutoEncoder import AutoEncoders
from utils.reports_generations import *
from utils.energy_analysis import *
import pandas as pd

#1,6,7,8,9,10,11,12,13,27,28,32,33,34,35,36,43,44
unsw = Dataset("C:\\Users\\slimanca\\Downloads\\archive\\UNSW_NB15_training-set.csv", [1,6,7,8,9,10,11,12,13,27,28,32,33,34,35,36,43,44], "C:\\Users\\slimanca\\Downloads\\archive\\UNSW_NB15_testing-set.csv", features_select=True)
x_train, x_test, y_train, y_test = unsw.preprocess(attack_label='label', attack_type_label='attack_cat', columns_to_encode=[], oversample=True, binarize_y=False)
rf_ids = IDSModel ("ES-RF", "RF", None)
rf_ids.explore_rf_models(x_train, y_train, x_test, y_test, None, None)
"""es_cnn = IDSModel("ES-CNN", "CNN", None)
es_cnn.load_ids_model("../../output/models/CNN/ES-CNN.h5")
es_cnn.generate_tflite_model(x_test, False)
structure = {
    'layers': [
        {'type': 'conv1d','params': {'filters': 64 ,'kernel_size' : 3, 'activation': "relu",'padding' :'same'}},
        {'type': 'conv1d','params': {'filters': 64 , 'kernel_size' : 3, 'activation': "relu",'padding' :'same'}},
        {'type': 'maxpool1d','params': {'pool_size': 2}},
        {'type': 'conv1d','params': {'filters': 128 , 'kernel_size' : 3, 'activation': "relu",'padding':'same'}},
        {'type': 'conv1d','params': {'filters': 128 , 'kernel_size' : 3, 'activation': "relu",'padding':'same'}},
        {'type': 'conv1d','params': {'filters': 128 , 'kernel_size' : 3, 'activation': "relu",'padding':'same'}},
        {'type': 'maxpool1d','params': {'pool_size': 2}},
        {'type': 'conv1d','params': {'filters': 256 , 'kernel_size' : 3, 'activation': "relu",'padding':'same'}},
        {'type': 'conv1d','params': {'filters': 256 , 'kernel_size' : 3, 'activation': "relu",'padding':'same'}},
        {'type': 'conv1d','params': {'filters': 256 , 'kernel_size' : 3, 'activation': "relu",'padding':'same'}},
        {'type': 'maxpool1d','params': {'pool_size': 2}},
        {'type': 'flatten', 'params': {}},
        {'type': 'dense', 'params': {'units': 100, 'kernel_initializer': 'glorot_uniform', 'activation': "relu"}},
        {'type': 'dropout', 'params': {'rate': 0.5}},
        {'type': 'dense', 'params': {'units': 20, 'kernel_initializer': 'glorot_uniform', 'activation': "relu"}},
        {'type': 'dense', 'params': {'units': 10, 'kernel_initializer': 'glorot_uniform', 'activation': "softmax"}},
    ]
}

ids = IDSModel("AE-CNN", "CNN", structure)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
ids.create_model(x_train, y_train, x_test, y_test)
ids.get_security_metrics(x_test, y_test)
ids.save_ids_model()
ids.generate_tflite_model(x_test, False)

unsw.write_test_data("x_test_unsw_AE")
#ids.explore_dnn_models(x_train, y_train, x_test, y_test, None, None)
#ids.load_ids_model("../../models/CNN/ES-")
#unsw.write_test_data("unsw_ES")
model = IDSModel("NoFS-DNN2", "DNN", None)
model.load_ids_model("../../output/models/DNN/NoFS-DNN2.h5")
acc, f1 = model.get_security_metrics(unsw.x_test, unsw.y_test)
print("Acc %f F1 %f" %(acc, f1))
#'proto', 'state', 'service'
#unsw.write_test_data("unsw_NoFS")
structure = {
    'layers': [
        {'type': 'dense','params': {'units': 1024 , 'kernel_initializer': 'glorot_uniform', 'activation': "relu"}},
        {'type': 'dense', 'params': {'units': 704, 'kernel_initializer': 'glorot_uniform', 'activation': "relu"}},
        {'type': 'dense', 'params': {'units': 288, 'kernel_initializer': 'glorot_uniform', 'activation': "relu"}},
        {'type': 'dense', 'params': {'units': 64, 'kernel_initializer': 'glorot_uniform', 'activation': "relu"}},
        {'type': 'dense', 'params': {'units': 10, 'kernel_initializer': 'glorot_uniform', 'activation': "softmax"}},

    ]
}


model = IDSModel("AE-DNN2", "DNN", structure)


model.create_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


model.save_ids_model()
#model.load_ids_model('../output/models/DNN/ES-DNN1.h5')80
acc, f1_score = model.get_security_metrics(x_test, y_test)
model.generate_tflite_model(x_test, opt=False)
impl1 = Implementation(model, None, 80000, acc, f1_score, None,  None, None, None, None)
"""
#read_and_plot_energy_from_file("../../output/energy_measures/nofs_dnn1_gpu_b512.csv")
