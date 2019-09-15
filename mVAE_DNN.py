"""
Ruifeng Hu
09132019
UTHealth-SBMI
"""

import os
import sys
import datetime
import time

import numpy as np
import pandas as pd

from keras import optimizers
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Lambda
from keras.losses import binary_crossentropy
import keras.backend as K
from keras.utils import plot_model

import tensorflow as tf
from tensorflow import set_random_seed

from sklearn.model_selection import train_test_split
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

#===================================================================
tissue = sys.argv[1]
# tissue = "Brain_Substantia_nigra"

Roadmap_encoded_df = pd.read_csv("encoded_Roadmap.txt", sep = "\t", index_col=0,header=0)
TF_encoded_df = pd.read_csv("encoded_TF.txt", sep = "\t", index_col=0,header=0)
DNAacc_encoded_df = pd.read_csv("encoded_DNAacc.txt", sep = "\t", index_col=0,header=0)

dataset_Y = Roadmap_encoded_df.iloc[:,[-1]]

Roadmap_encoded_df.drop(['y_true'], axis=1, inplace=True)
TF_encoded_df.drop(['y_true'], axis=1, inplace=True)
DNAacc_encoded_df.drop(['y_true'], axis=1, inplace=True)

# make the input dataframe
frames = [Roadmap_encoded_df, TF_encoded_df, DNAacc_encoded_df, dataset_Y]
dataset_input = pd.concat(frames, axis=1, sort=False)
print(dataset_input.shape)

X_dataset = dataset_input.drop('y_true', axis=1)
y_dataset = dataset_input['y_true']
print(X_dataset.shape)

X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, test_size=0.20)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20)

batch_size = 16
epoach = 15
learning_rate = 0.005

input_dim_x = X_train.shape[1]
# model = Sequential()
# model.add(Dense(64, input_dim=input_dim_x, activation='relu', kernel_initializer='random_uniform'))
# model.add(Dense(16, activation='relu', kernel_initializer='random_uniform'))
# model.add(Dense(4, kernel_initializer='random_uniform'))
# model.add(Dense(1, activation='sigmoid'))

inputs = Input(shape=(input_dim_x,), name="mVAE_DNN_Input")
x = Dense(64, activation='relu', kernel_initializer='random_uniform', name="mVAE_DNN_Dense1")(inputs)
x = Dense(16, activation='relu', kernel_initializer='random_uniform', name="mVAE_DNN_Dense2")(x)
x = Dense(4, kernel_initializer='random_uniform', name="mVAE_DNN_Dense3")(x)
x = Dense(1, activation='sigmoid',name="mVAE_DNN_output")(x)

model = Model(inputs=inputs, outputs=x)

adam = optimizers.Adam(lr=learning_rate)

model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['binary_accuracy', f1_m, precision_m, recall_m])

result = model.fit(np.array(X_train, dtype=np.float32),
                   np.array(y_train, dtype=np.float32),
                   epochs=epoach,
                   batch_size=batch_size,
                   validation_data=(np.array(X_val, dtype=np.float32),
                                    np.array(y_val, dtype=np.float32)),
                   verbose=1)

y_pred_x = model.predict(np.array(X_test))
y_pred = y_pred_x > 0.5

accuracy_test = round(accuracy_score(np.array(y_test, dtype=np.float32), y_pred), 4)
precision_test = round(precision_score(np.array(y_test, dtype=np.float32), y_pred), 4)
recall_test = round(recall_score(np.array(y_test, dtype=np.float32), y_pred), 4)

AUROC_test = round(roc_auc_score(np.array(y_test, dtype=np.float32), y_pred_x),4)
average_precision = round(average_precision_score(y_test, y_pred_x), 4)

fpr_roc, tpr_roc, thresholds_roc = roc_curve(y_test, y_pred_x)
precision_prc, recall_prc, thresholds_prc = precision_recall_curve(y_test, y_pred_x)

del model
K.clear_session()

#return [accuracy_test, precision_test,recall_test, AUROC_test, average_precision, fpr_roc, tpr_roc,precision_prc, recall_prc]
