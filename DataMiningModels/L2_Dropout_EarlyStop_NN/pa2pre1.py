#    name: pa2pre.py
# purpose: Student's add code to preprocessing of the data

# Recall that any preprocessing you do on your training
# data, you must also do on any future data you want to
# predict.  This file allows you to perform any
# preprocessing you need on my undisclosed test data

NB_CLASSES=10
import numpy as np
from keras import utils as np_utils
from sklearn import preprocessing

def processTestData(X, y):

    X = X.reshape((X.shape[0], X.shape[1]*X.shape[2]))
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)

    # y preprocessing goes here.  y_test becomes a ohe
    y_ohe = np_utils.to_categorical (y, NB_CLASSES)
    return X, y_ohe