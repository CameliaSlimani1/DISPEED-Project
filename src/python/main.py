from entities.Platform import Platforme
from entities.Implementation import Implementation
from entities.IDSModel import IDSModel
from entities.Dataset import Dataset
from entities.AutoEncoder import AutoEncoders
from utils.reports_generations import *
import pandas as pd

unsw = Dataset("C:\\Users\\slimanca\\Downloads\\archive\\UNSW_NB15_training-set.csv", [], "C:\\Users\\slimanca\\Downloads\\archive\\UNSW_NB15_testing-set.csv", features_select=False)
x_train, x_test, y_train, y_test = unsw.preprocess(attack_label='label', attack_type_label='attack_cat', columns_to_encode=['proto', 'state', 'service'], oversample=True, binarize_y=True)

structure = {
    'layers': [
        {'type': 'dense','params': {'units': 128 , 'kernel_initializer': 'glorot_uniform', 'activation': "relu"}},
        {'type': 'dropout', 'params': {'rate': 0.5}},
        {'type': 'dense', 'params': {'units': 64, 'kernel_initializer': 'glorot_uniform', 'activation': "relu"}},
        {'type': 'dense', 'params': {'units': 32, 'kernel_initializer': 'glorot_uniform', 'activation': "relu"}},
        {'type': 'dense', 'params': {'units': 10, 'kernel_initializer': 'glorot_uniform', 'activation': "softmax"}},

    ]
}


model = IDSModel("NoFS-DNN1", "DNN", structure)


model.create_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


model.save_ids_model()
#model.load_ids_model('../output/models/DNN/ES-DNN1.h5')
acc, f1_score = model.get_security_metrics(x_test, y_test)
model.generate_tflite_model(x_test, opt=False)
impl1 = Implementation(model, None, 8000, acc, f1_score, None,  None, None, None, None)
