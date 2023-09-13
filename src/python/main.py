from entities.Platform import Platforme
from entities.Implementation import Implementation
from entities.IDSModel import IDSModel
from entities.Dataset import Dataset
from entities.AutoEncoder import AutoEncoders
from utils.reports_generations import *
import pandas as pd

#1,6,7,8,9,10,11,12,13,27,28,32,33,34,35,36,43,44
unsw = Dataset("C:\\Users\\slimanca\\Downloads\\archive\\UNSW_NB15_training-set.csv", [1,6,7,8,9,10,11,12,13,27,28,32,33,34,35,36,43,44], "C:\\Users\\slimanca\\Downloads\\archive\\UNSW_NB15_testing-set.csv", features_select=True)
x_train, x_test, y_train, y_test = unsw.preprocess(attack_label='label', attack_type_label='attack_cat', columns_to_encode=[], oversample=True, binarize_y=True)

ids = IDSModel("dnn", "DNN", None)
ids.explore_dnn_models(x_train, y_train, x_test, y_test, None, None)

#unsw.write_test_data("unsw_ES")
"""model = IDSModel("NoFS-DNN2", "DNN", None)
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
#model.load_ids_model('../output/models/DNN/ES-DNN1.h5')
acc, f1_score = model.get_security_metrics(x_test, y_test)
model.generate_tflite_model(x_test, opt=False)
impl1 = Implementation(model, None, 80000, acc, f1_score, None,  None, None, None, None)
"""