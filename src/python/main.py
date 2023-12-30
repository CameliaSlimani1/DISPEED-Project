from entities.Platform import Platforme
from entities.Implementation import Implementation
from entities.IDSModel import IDSModel
from entities.Dataset import Dataset
from entities.AutoEncoder import AutoEncoders
from utils.reports_generations import *
from utils.energy_analysis import *
import pandas as pd

#exploration = pd.read_json("../../output/explorations/ES-RF_pareto.json")
#print(exploration[["accuracy", "size"]])

#1,6,7,8,9,10,11,12,13,27,28,32,33,34,35,36,43,44
unsw = Dataset("C:\\Users\\slimanca\\Downloads\\archive\\UNSW_NB15_training-set.csv", [], "C:\\Users\\slimanca\\Downloads\\archive\\UNSW_NB15_testing-set.csv", features_select=False)
x_train, x_test, y_train, y_test = unsw.preprocess(attack_label='label', attack_type_label='attack_cat', columns_to_encode=['proto', 'service', 'state'], oversample=True, binarize_y=False)

structure = {'n_estimators' : 5, 'max_depth' : 100}

rf = IDSModel("NoFS-RF", 'RF', structure)

rf.create_model(x_train, y_train, x_test, y_test)
rf.generate_rf_for_cpu(x_train, x_test)



