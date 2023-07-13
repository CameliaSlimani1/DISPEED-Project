import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import csv

class Dataset:



    def __init__(self, path_to_trainset, features, path_to_testset=None, features_select=False ):
        self.path_to_train_set = path_to_trainset
        self.path_to_testset = path_to_testset
        self.attack_label = None
        self.attack_type_label = None
        self.columns_to_encode = None
        self.oversample = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        #Read data
        self.data = pd.read_csv(open(path_to_trainset))
        if self.path_to_testset != None :
            test_data = pd.read_csv(open(self.path_to_testset))
            self.data = pd.concat([self.data, test_data], axis=0)

        # Explicit selection of features
        if features_select :
            self.data = self.data.iloc[:, features]

    def preprocess(self, attack_label, attack_type_label, columns_to_encode, oversample=True, binarize_y=True):
        self.attack_label = attack_label
        self.attack_type_label = attack_type_label
        self.columns_to_encode = columns_to_encode
        self.oversample = oversample

        # Replace NaNs by 'normal' for attack_type_label column and the other nan values by 0
        self.data[self.attack_type_label] = self.data[self.attack_type_label].replace(np.nan, 'Normal')
        self.data = self.data.fillna(0)

        # Extract class label to form y

        y_attack = self.data[self.attack_label]
        y_attack_type = self.data[self.attack_type_label]

        # Encode class label
        le = LabelEncoder()
        le.fit(y_attack_type)
        y = le.transform(y_attack_type)
        self.data = self.data.drop(columns=[self.attack_label, self.attack_type_label])

        # One-hot encode categorical columns
        self.data = pd.get_dummies(self.data, prefix=self.columns_to_encode, columns=self.columns_to_encode)

        # Split data into train and test data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data, y, random_state=123, test_size=0.3)

        # Oversample data if oversample argument = True
        if oversample:
            oversampler = SMOTE()
            self.x_train = self.x_train.astype("float32")
            self.x_train, self.y_train = oversampler.fit_resample(self.x_train, self.y_train)

        # Scale data using MinMax

        scaler = MinMaxScaler()
        scaler.fit(self.x_train)
        self.x_train = scaler.transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)

         # Binarize y_train and y_test incase of CNN/DNN model
        if binarize_y :
            lb = LabelBinarizer()
            lb.fit(y)
            self.y_train = lb.transform(self.y_train)
            self.y_test = lb.transform(self.y_test)
        return self.x_train, self.x_test, self.y_train, self.y_test

    def write_test_data(self, filename):
        np.savetxt(f"../../output/test_data/x_test_%s.csv" % filename, self.x_test, delimiter=",")
        np.savetxt(f"../../output/test_data/y_test_%s.csv" % filename, np.argmax(self.y_test, axis=1), delimiter=",")
