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

##======================================================================
tissue = sys.argv[1]
file = sys.argv[2]

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

X_train, X_val, y_train, y_val = train_test_split(dataset_input_X, y_true, test_size=0.20)

val_index = X_val.index
train_index = X_train.index

dataset_val_X = dataset_input_X.loc[val_index,:]
dataset_train_X = dataset_input_X.loc[train_index, :]

dataset_val_Y = y_true.loc[val_index,:]
dataset_train_Y = y_true.loc[train_index, :]

print("Train:", dataset_train_X.shape)
print("Val:", dataset_val_X.shape)

dataset_X = [dataset_train_X, dataset_val_X]
dataset_Y = [dataset_train_Y, dataset_val_Y]
#=======================================================================

dataset_train_X = dataset_X[0]
dataset_val_X = dataset_X[1]

dataset_train_Y = dataset_Y[0]
dataset_val_Y = dataset_Y[1]

# 1.Roadmap
Roadmap_input_X = dataset_input_X.iloc[:, dataset_input_X.columns.str.startswith('Roadmap_')]

Roadmap_train_X = dataset_train_X.iloc[:, dataset_train_X.columns.str.startswith('Roadmap_')]
Roadmap_val_X = dataset_val_X.iloc[:, dataset_val_X.columns.str.startswith('Roadmap_')]

# 2.TF
TF_input_X = dataset_input_X.iloc[:, dataset_input_X.columns.str.startswith('TF_')]

TF_train_X = dataset_train_X.iloc[:, dataset_train_X.columns.str.startswith('TF_')]
TF_val_X = dataset_val_X.iloc[:, dataset_val_X.columns.str.startswith('TF_')]

# 3.DNAacc
DNAacc_input_X = dataset_input_X.iloc[:, dataset_input_X.columns.str.startswith('DNAacc_')]

DNAacc_train_X = dataset_train_X.iloc[:, dataset_train_X.columns.str.startswith('DNAacc_')]
DNAacc_val_X = dataset_val_X.iloc[:, dataset_val_X.columns.str.startswith('DNAacc_')]

#======================================================================
def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=tf.shape(z_mean))
    return z_mean + K.exp(z_log_sigma * 0.5) * epsilon

def vae_loss(x, x_decoded_mean):
    xent_loss = original_dim * binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    return K.mean(xent_loss + kl_loss)
# ===========================================================
learning_rate = 0.005
batch_size_AE = 32
epochs_AE = 15

## Roadmap
print("[INFO] Roadmap AE running...")
original_dim = Roadmap_train_X.shape[1]
intermediate_dim = 256
latent_dim = 64

x = Input(shape=(original_dim,), name="mVAE_Roadmap_input")
h = Dense(intermediate_dim, activation='relu', name="mVAE_Roadmap_hidden")(x)
z_mean = Dense(latent_dim,name="mVAE_Roadmap_zmean")(h)
z_log_sigma = Dense(latent_dim,name="mVAE_Roadmap_zlogsigma")(h)

z = Lambda(sampling, output_shape=(latent_dim,),name="mVAE_Roadmap_z")([z_mean, z_log_sigma])

h_decoded = Dense(intermediate_dim, activation='relu', name="mVAE_Roadmap_hdecode")(z)
x_decoded_mean = Dense(original_dim, activation='sigmoid',name="mVAE_Roadmap_decodemean")(h_decoded)

Roadmap_VAE = Model(x, x_decoded_mean)
Roadmap_encoder = Model(x, z_mean)

adam = optimizers.Adam(lr=learning_rate)
Roadmap_VAE.compile(loss=vae_loss, optimizer=adam)

Roadmap_res = Roadmap_VAE.fit(Roadmap_train_X, Roadmap_train_X, epochs=epochs_AE,
                              batch_size=batch_size_AE,
                              validation_data=(Roadmap_val_X, Roadmap_val_X),
                              verbose=1)

encoded_df = Roadmap_encoder.predict(Roadmap_input_X)
Roadmap_encoded_df = pd.DataFrame(encoded_df, index=Roadmap_input_X.index)
frames = [Roadmap_encoded_df, y_true]
Roadmap_df = pd.concat(frames, axis=1, sort=False)

encoded_file = os.path.join('', 'encoded_Roadmap.txt')
Roadmap_df.to_csv(encoded_file, sep='\t', index=True)

## TF
print("[INFO] TF AE running...")
original_dim = TF_train_X.shape[1]
intermediate_dim = 256
latent_dim = 64

x = Input(shape=(original_dim,), name="mVAE_TF_input")
h = Dense(intermediate_dim, activation='relu', name="mVAE_TF_hidden")(x)
z_mean = Dense(latent_dim, name="mVAE_TF_zmean")(h)
z_log_sigma = Dense(latent_dim, name="mVAE_TF_zlogsigma")(h)

z = Lambda(sampling, output_shape=(latent_dim,), name="mVAE_TF_z")([z_mean, z_log_sigma])

h_decoded = Dense(intermediate_dim, activation='relu', name="mVAE_TF_hdecode")(z)
x_decoded_mean = Dense(original_dim, activation='sigmoid', name="mVAE_TF_decodemean")(h_decoded)

TF_VAE = Model(x, x_decoded_mean)
TF_encoder = Model(x, z_mean)

adam = optimizers.Adam(lr=learning_rate)
TF_VAE.compile(loss=vae_loss, optimizer=adam)

TF_res = TF_VAE.fit(TF_train_X, TF_train_X, epochs=epochs_AE,
                    batch_size=batch_size_AE,
                    validation_data=(TF_val_X, TF_val_X),
                    verbose=1)

encoded_df = TF_encoder.predict(TF_input_X)
TF_encoded_df = pd.DataFrame(encoded_df, index=TF_input_X.index)
frames = [TF_encoded_df, y_true]
TF_df = pd.concat(frames, axis=1, sort=False)

encoded_file = os.path.join('', 'encoded_TF.txt')
TF_df.to_csv(encoded_file, sep='\t', index=True)

## DNAacc
print("[INFO] DNAacc AE running...")
original_dim = DNAacc_train_X.shape[1]
intermediate_dim = 256
latent_dim = 64

x = Input(shape=(original_dim,),name="mVAE_DNAacc_Input")
h = Dense(intermediate_dim, activation='relu',name="mVAE_DNAacc_hidden")(x)
z_mean = Dense(latent_dim,name="mVAE_DNAacc_zmean")(h)
z_log_sigma = Dense(latent_dim,name="mVAE_DNAacc_zlogsigma")(h)

z = Lambda(sampling, output_shape=(latent_dim,),name="mVAE_DNAacc_z")([z_mean, z_log_sigma])

h_decoded = Dense(intermediate_dim, activation='relu',name="mVAE_DNAacc_hdecode")(z)
x_decoded_mean = Dense(original_dim, activation='sigmoid',name="mVAE_DNAacc_decodemean")(h_decoded)

DNAacc_VAE = Model(x, x_decoded_mean)
DNAacc_encoder = Model(x, z_mean)

adam = optimizers.Adam(lr=learning_rate)
DNAacc_VAE.compile(loss=vae_loss, optimizer=adam)

DNAacc_res = DNAacc_VAE.fit(DNAacc_train_X, DNAacc_train_X, epochs=epochs_AE,
                            batch_size=batch_size_AE,
                            validation_data=(DNAacc_val_X, DNAacc_val_X),
                            verbose=1)

encoded_df = DNAacc_encoder.predict(DNAacc_input_X)
DNAacc_encoded_df = pd.DataFrame(encoded_df, index=DNAacc_input_X.index)
frames = [DNAacc_encoded_df, y_true]
DNAacc_df = pd.concat(frames, axis=1, sort=False)

encoded_file = os.path.join('', 'encoded_DNAacc.txt')
DNAacc_df.to_csv(encoded_file, sep='\t', index=True)
