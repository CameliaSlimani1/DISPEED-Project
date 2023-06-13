from entities.Platform import Platforme
from entities.Implementation import Implementation
from entities.IDSModel import IDSModel
from entities.Dataset import Dataset
from entities.AutoEncoder import AutoEncoders
from utils.reports_generations import *
import pandas as pd

platform= Platforme.getPlatformByName()
unsw = Dataset("C:\\Users\\slimanca\\Downloads\\archive\\UNSW_NB15_training-set.csv", [1,6,7,8,9,10,11,12,13,27,28,32,33,34,35,36,43,44], "C:\\Users\\slimanca\\Downloads\\archive\\UNSW_NB15_testing-set.csv", features_select=False)
x_train, x_test, y_train, y_test = unsw.preprocess(attack_label='label', attack_type_label='attack_cat', columns_to_encode=['proto', 'service', 'state'],oversample=True, binarize_y=True)
encoder = AutoEncoders(197,25, False)
x_train = encoder.encoder.predict(x_train)
x_test = encoder.encoder.predict(x_test)
structure = {
    'layers': [
        {'type': 'conv1d', 'params': {'filters': 64, 'kernel_size': 3,'kernel_initializer':'glorot_uniform',  'padding': 'same', 'activation': "relu", 'input_shape': (x_train.shape[1],1)}},
        {'type': 'conv1d', 'params': {'filters': 64, 'kernel_size': 3, 'kernel_initializer': 'glorot_uniform', 'padding': 'same', 'activation': "relu"}},
        {'type': 'maxpool1d', 'params': {'pool_size':2}},
        {'type': 'conv1d','params': {'filters': 128, 'kernel_size': 3, 'kernel_initializer': 'glorot_uniform', 'padding': 'same', 'activation': "relu"}},
        {'type': 'conv1d','params': {'filters': 128, 'kernel_size': 3, 'kernel_initializer': 'glorot_uniform', 'padding': 'same', 'activation': "relu"}},
        {'type': 'conv1d','params': {'filters': 128, 'kernel_size': 3, 'kernel_initializer': 'glorot_uniform', 'padding': 'same', 'activation': "relu"}},
        {'type': 'maxpool1d', 'params': {'pool_size': 2}},
        {'type': 'conv1d', 'params': {'filters': 256, 'kernel_size': 3, 'kernel_initializer': 'glorot_uniform','padding': 'same', 'activation': "relu"}},
        {'type': 'conv1d', 'params': {'filters': 256, 'kernel_size': 3, 'kernel_initializer': 'glorot_uniform','padding': 'same', 'activation': "relu"}},
        {'type': 'conv1d', 'params': {'filters': 256, 'kernel_size': 3, 'kernel_initializer': 'glorot_uniform','padding': 'same', 'activation': "relu"}},
        {'type': 'maxpool1d', 'params': {'pool_size': 2}},
        {'type': 'flatten', 'params': {}},
        {'type': 'dense','params': {'units': 100, 'kernel_initializer': 'glorot_uniform', 'activation': "relu"}},
        {'type': 'dropout', 'params': {'rate': 0.5}},
        {'type': 'dense', 'params': {'units': 20, 'kernel_initializer': 'glorot_uniform', 'activation': "relu"}},
        {'type': 'dense', 'params': {'units': 10, 'kernel_initializer': 'glorot_uniform', 'activation': "softmax"}},

    ]
}

model = IDSModel("AE-CNN", "CNN", structure)

model.create_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


model.save_ids_model()
acc, f1_score = model.get_security_metrics(x_test, y_test)
impl1 = Implementation(model, platform, 8000, acc, f1_score, 140,  112.1, 26264, 320, "test")
impl1.serialize()

print(impl1)