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
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, concatenate
import keras.backend as K
from keras.utils import plot_model

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

def get_model(input_dim, str):
    '''
    Build and return a DNN model
    :param input_dim:
    :return: DNN nodel
    '''

    inputs = Input(shape=(input_dim,), name=str + '_input')

    x = Dense(512, activation='relu', kernel_initializer='random_uniform', name=str + '_sub_Dense1')(inputs)
    x = Dropout(0.2)(x)  # Avoid overfitting

    x = Dense(256, activation='relu', kernel_initializer='random_uniform', name=str + '_sub_Dense2')(x)
    x = Dropout(0.2)(x)

    x = Dense(128, activation='relu', kernel_initializer='random_uniform', name=str + '_sub_Dense3')(x)
    x = Dropout(0.2)(x)

    x = Dense(64, activation='relu', kernel_initializer='random_uniform', name=str + '_sub_Dense4')(x)

    model = Model(inputs=inputs, outputs=x)

    return model

#========================================================================================
tissue = sys.argv[1]
file = sys.argv[2]
Results = sys.argv[3]

#tissue = "Brain_Substantia_nigra"
# path = "C:\\Users\\hurui\\Dropbox\\Temp\\"
# file = path + tissue+'_matrix_1v3_signal_100.txt'

## load dataset
print('[INFO] Loading '+tissue+' data...')
print(file)
dataset = pd.read_csv(file, sep='\t',index_col=0, header=0)
print(dataset.shape)

## Subset: selecte dataset
print("Subset:")
y_true = dataset.iloc[:, [-1]]
GC = dataset.loc[:, ["GC"]]

# 1.Roadmap
print("Roadmap:")
dataset_Roadmap = dataset.iloc[:, dataset.columns.str.startswith("Roadmap_")]
print(dataset_Roadmap.shape)
dataset_Roadmap = dataset_Roadmap.loc[:, (dataset_Roadmap != 0).any(axis=0)]
print(dataset_Roadmap.shape)

# 2.TF
print("TF:")
dataset_TF = dataset.iloc[:, dataset.columns.str.startswith("TF_")]
print(dataset_TF.shape)
dataset_TF = dataset_TF.loc[:, (dataset_TF != 0).any(axis=0)]
print(dataset_TF.shape)

# 3.DNAacc
print("DNAacc:")
dataset_DNAacc = dataset.iloc[:, dataset.columns.str.startswith("DNAacc_")]
print(dataset_DNAacc.shape)
dataset_DNAacc = dataset_DNAacc.loc[:, (dataset_DNAacc != 0).any(axis=0)]
print(dataset_DNAacc.shape)

# 4.Methyl
print("Methyl:")
dataset_Methyl = dataset.iloc[:, dataset.columns.str.startswith("Methyl_")]
print(dataset_Methyl.shape)
dataset_Methyl = dataset_Methyl.loc[:, (dataset_Methyl != 0).any(axis=0)]
print(dataset_Methyl.shape)

# Make the input dataframe
frames = [dataset_Roadmap, dataset_TF, dataset_DNAacc]
dataset_input_X = pd.concat(frames, axis=1, sort=False)
print("Input Shape:",dataset_input_X.shape)

print("[INFO] Sorting data ...")
dataset_input_X.sort_index(inplace=True)

print("[INFO] Normalizing data ...")
dataset_input_X = dataset_input_X.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
print(dataset_input_X.shape)

print("[INFO] Drop features that caontain NA values of data...")
dataset_input_X.dropna(axis="columns", inplace=True)
print(dataset_input_X.shape)

X_train, X_test, y_train, y_test = train_test_split(dataset_input_X, y_true, test_size=0.20)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20)

test_index = X_test.index
val_index = X_val.index
train_index = X_train.index

dataset_test_X = dataset_input_X.loc[test_index,:]
dataset_val_X = dataset_input_X.loc[val_index,:]
dataset_train_X = dataset_input_X.loc[train_index, :]

dataset_test_Y = y_true.loc[test_index, :]
dataset_val_Y = y_true.loc[val_index,:]
dataset_train_Y = y_true.loc[train_index, :]

print("Train:", dataset_train_X.shape)
print("Val:", dataset_val_X.shape)
print("Test:", dataset_test_X.shape)

dataset_X = [dataset_train_X, dataset_val_X, dataset_test_X]
dataset_Y = [dataset_train_Y, dataset_val_Y, dataset_test_Y]

dataset_train_X = dataset_X[0]
dataset_val_X = dataset_X[1]
dataset_test_X = dataset_X[2]

# 1.Roadmap
Roadmap_train_X = dataset_train_X.iloc[:, dataset_train_X.columns.str.startswith('Roadmap_')]
Roadmap_val_X = dataset_val_X.iloc[:, dataset_val_X.columns.str.startswith('Roadmap_')]
Roadmap_test_X = dataset_test_X.iloc[:, dataset_test_X.columns.str.startswith('Roadmap_')]


# 2.TF
TF_train_X = dataset_train_X.iloc[:, dataset_train_X.columns.str.startswith('TF_')]
TF_val_X = dataset_val_X.iloc[:, dataset_val_X.columns.str.startswith('TF_')]
TF_test_X = dataset_test_X.iloc[:, dataset_test_X.columns.str.startswith('TF_')]


# 3.DNAacc
DNAacc_train_X = dataset_train_X.iloc[:, dataset_train_X.columns.str.startswith('DNAacc_')]
DNAacc_val_X = dataset_val_X.iloc[:, dataset_val_X.columns.str.startswith('DNAacc_')]
DNAacc_test_X = dataset_test_X.iloc[:, dataset_test_X.columns.str.startswith('DNAacc_')]

dataset_train_Y = dataset_Y[0]
dataset_val_Y = dataset_Y[1]
dataset_test_Y = dataset_Y[2]

batch_size = 32
epoach = 12
learning_rate = 0.005

##=========================================================
Roadmap_model = get_model(Roadmap_train_X.shape[1], 'Roadmap')
TF_model = get_model(TF_train_X.shape[1], 'TF')
DNAacc_model = get_model(DNAacc_train_X.shape[1], 'DNAacc')

concat_frames = [Roadmap_model.output, TF_model.output, DNAacc_model.output]
model_input = [Roadmap_model.input, TF_model.input, DNAacc_model.input]

train_sets = [Roadmap_train_X, TF_train_X, DNAacc_train_X]
val_sets = [Roadmap_val_X, TF_val_X, DNAacc_val_X]
test_sets = [Roadmap_test_X, TF_test_X, DNAacc_test_X]

combined_model = concatenate(concat_frames)
z = Dense(64, activation='relu', kernel_initializer='random_uniform', name='mDNN_Dense1')(combined_model)
z = Dense(16, activation='relu', kernel_initializer='random_uniform', name='mDNN_Dense2')(z)
z = Dense(4, kernel_initializer='random_uniform', name='mDNN_Dense3')(z)
z = Dense(1, activation='sigmoid', name='mDNN_output')(z)

model = Model(inputs=model_input, outputs=z)

adam = optimizers.Adam(lr=learning_rate)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['binary_accuracy', f1_m, precision_m, recall_m])

# output_model_file = os.path.join('', 'mDNN_architecture_' + st + '.png')
# plot_model(model, to_file=output_model_file)

result = model.fit(train_sets,
                   dataset_train_Y,
                   epochs=epoach,
                   batch_size=batch_size,
                   validation_data=(val_sets, dataset_val_Y),
                   verbose=1)

y_pred_x = model.predict(test_sets)
y_pred = y_pred_x > 0.5

accuracy_test = accuracy_score(np.array(dataset_test_Y, dtype=np.float32), y_pred)
precision_test = precision_score(np.array(dataset_test_Y, dtype=np.float32), y_pred)
recall_test = recall_score(np.array(dataset_test_Y, dtype=np.float32), y_pred)

AUROC_test = roc_auc_score(np.array(dataset_test_Y, dtype=np.float32), y_pred_x)
average_precision = average_precision_score(dataset_test_Y, y_pred_x)

fpr_roc, tpr_roc, thresholds_roc = roc_curve(dataset_test_Y, y_pred_x)
precision_prc, recall_prc, thresholds_prc = precision_recall_curve(dataset_test_Y, y_pred_x)

del Roadmap_model
del TF_model
del DNAacc_model
del model
K.clear_session()

data_x = { "accuracy": np.array([accuracy_test, precision_test, recall_test, AUROC_test,average_precision]),
           "fpr": fpr_roc,
           "tpr": tpr_roc,
           "precision":precision_prc,
           "recall":recall_prc}
np.savez(Results+"/"+tissue+'-mDNN.npz', **data_x)

#return [accuracy_test, precision_test,recall_test, AUROC_test, average_precision, fpr_roc, tpr_roc,precision_prc, recall_prc]
