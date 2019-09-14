"""
Ruifeng Hu
09132019
UTHealth-SBMI
"""

#from ..utils.functions import acc_dist

import os
import datetime
import time

import numpy as np

from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras.backend as K

from tensorflow import set_random_seed

from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S")

#==========================================================
## fix random seed for reproducibility
np.random.seed(12)
set_random_seed(12)

os.environ["KERAS_BACKEND"] = "tensorflow"
# The GPU id to use, usually either "0" , "1", "2" or "3" if there are 4 GPUs;
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#==========================================================
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def eQTL_loss(y_true, y_pred):
    if y_true == 0:
        y_pred = y_pred * 1.2
    if y_true == 1:
        y_pred = y_pred / 1.2

    return K.sum(K.abs(y_true - y_pred))

def DNN(dataset_X,dataset_Y):
    dataset_train_X = dataset_X[0]
    dataset_val_X = dataset_X[1]
    dataset_test_X = dataset_X[2]

    dataset_train_Y = dataset_Y[0]
    dataset_val_Y = dataset_Y[1]
    dataset_test_Y = dataset_Y[2]


    batch_size = 32
    epoach = 10
    learning_rate = 0.005

    input_dim_x = dataset_train_X.shape[1]
    model = Sequential()
    model.add(Dense(1024, input_dim=input_dim_x,activation='relu', kernel_initializer='random_uniform'))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu', kernel_initializer='random_uniform'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu', kernel_initializer='random_uniform'))
    model.add(Dropout(0.25))
    model.add(Dense(16, activation='relu', kernel_initializer='random_uniform'))
    model.add(Dense(4, kernel_initializer='random_uniform'))
    model.add(Dense(1, activation='sigmoid'))

    adam = optimizers.Adam(lr=learning_rate)

    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['binary_accuracy', f1_m, precision_m, recall_m])

    result = model.fit(np.array(dataset_train_X, dtype=np.float32),
                       np.array(dataset_train_Y, dtype=np.float32),
                       epochs=epoach,
                       batch_size=batch_size,
                       validation_data=(np.array(dataset_val_X, dtype=np.float32),
                                        np.array(dataset_val_Y, dtype=np.float32)),
                       verbose=1)

    y_pred_x = model.predict(np.array(dataset_test_X))
    y_pred = y_pred_x > 0.5

    accuracy_test = round(accuracy_score(np.array(dataset_test_Y, dtype=np.float32), y_pred), 4)
    precision_test = round(precision_score(np.array(dataset_test_Y, dtype=np.float32), y_pred), 4)
    recall_test = round(recall_score(np.array(dataset_test_Y, dtype=np.float32), y_pred), 4)

    AUROC_test = roc_auc_score(np.array(dataset_test_Y, dtype=np.float32), y_pred_x)
    average_precision = round(average_precision_score(dataset_test_Y, y_pred_x), 4)

    fpr_roc, tpr_roc, thresholds_roc = roc_curve(dataset_test_Y, y_pred_x)
    precision_prc, recall_prc, thresholds_prc = precision_recall_curve(dataset_test_Y, y_pred_x)

    del model
    K.clear_session()

    return [accuracy_test, precision_test,recall_test, AUROC_test, average_precision, fpr_roc, tpr_roc,precision_prc, recall_prc]



