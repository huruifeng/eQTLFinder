"""
Ruifeng Hu
09132019
UTHealth-SBMI
"""

import os
import datetime
import time

import numpy as np
import pandas as pd

from keras import optimizers
from keras.models import Model
from keras.layers import Dense, Input, Lambda
from keras.losses import binary_crossentropy
import keras.backend as K
from keras.utils import plot_model

import tensorflow as tf
from tensorflow import set_random_seed
from sklearn.linear_model import LogisticRegression

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


def multiVAE_LR(dataset_X,dataset_Y):


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

    # ===========================================================
    learning_rate = 0.005
    batch_size_AE = 16
    epochs_AE = 15

    ## Roadmap
    print("[INFO] Roadmap AE running...")
    original_dim = Roadmap_train_X.shape[1]
    intermediate_dim = 256
    latent_dim = 64

    x = Input(shape=(original_dim,), name="mVAE_LR_Roadmap_input")
    h = Dense(intermediate_dim, activation='relu', name="mVAE_LR_Roadmap_hidden")(x)
    z_mean = Dense(latent_dim, name="mVAE_LR_Roadmap_zmean")(h)
    z_log_sigma = Dense(latent_dim, name="mVAE_LR_Roadmap_zlogsigma")(h)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=tf.shape(z_mean))
        return z_mean + K.exp(z_log_sigma * 0.5) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,), name="mVAE_LR_Roadmap_z")([z_mean, z_log_sigma])

    h_decoded = Dense(intermediate_dim, activation='relu', name="mVAE_LR_Roadmap_hdecode")(z)
    x_decoded_mean = Dense(original_dim, activation='sigmoid', name="mVAE_LR_Roadmap_decodemean")(h_decoded)

    Roadmap_VAE = Model(x, x_decoded_mean)
    Roadmap_encoder = Model(x, z_mean)

    def vae_loss(x, x_decoded_mean):
        xent_loss = original_dim * binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
        return K.mean(xent_loss + kl_loss)

    adam = optimizers.Adam(lr=learning_rate)
    Roadmap_VAE.compile(loss=vae_loss, optimizer=adam)

    output_model_file = os.path.join('', 'Roadmap_VAE_LR_architecture_' + st + '.png')
    plot_model(Roadmap_VAE, to_file=output_model_file)

    Roadmap_res = Roadmap_VAE.fit(Roadmap_train_X, Roadmap_train_X, epochs=epochs_AE,
                                  batch_size=batch_size_AE,
                                  validation_data=(Roadmap_val_X, Roadmap_val_X),
                                  verbose=1)

    encoded_df = Roadmap_encoder.predict(Roadmap_train_X)
    Roadmap_encoded_df = pd.DataFrame(encoded_df, index=Roadmap_train_X.index)

    encoded_file = os.path.join('', 'encoded_Roadmap.txt')
    Roadmap_encoded_df.to_csv(encoded_file, sep='\t', index=True)

    ## TF
    print("[INFO] TF AE running...")
    original_dim = TF_train_X.shape[1]
    intermediate_dim = 256
    latent_dim = 64

    x = Input(shape=(original_dim,), name="mVAE_LR_TF_input")
    h = Dense(intermediate_dim, activation='relu', name="mVAE_LR_TF_hidden")(x)
    z_mean = Dense(latent_dim, name="mVAE_LR_TF_zmean")(h)
    z_log_sigma = Dense(latent_dim, name="mVAE_LR_TF_zlogsigma")(h)

    z = Lambda(sampling, output_shape=(latent_dim,), name="mVAE_LR_TF_z")([z_mean, z_log_sigma])

    h_decoded = Dense(intermediate_dim, activation='relu', name="mVAE_LR_TF_hdecode")(z)
    x_decoded_mean = Dense(original_dim, activation='sigmoid', name="mVAE_LR_TF_decodemean")(h_decoded)

    TF_VAE = Model(x, x_decoded_mean)
    TF_encoder = Model(x, z_mean)

    adam = optimizers.Adam(lr=learning_rate)
    TF_VAE.compile(loss=vae_loss, optimizer=adam)

    output_model_file = os.path.join('', 'TF_VAE_LR_architecture_' + st + '.png')
    plot_model(TF_VAE, to_file=output_model_file)

    TF_res = TF_VAE.fit(TF_train_X, TF_train_X, epochs=epochs_AE,
                        batch_size=batch_size_AE,
                        validation_data=(TF_val_X, TF_val_X),
                        verbose=1)

    encoded_df = TF_encoder.predict(TF_train_X)
    TF_encoded_df = pd.DataFrame(encoded_df, index=TF_train_X.index)

    encoded_file = os.path.join('', 'encoded_TF.txt')
    TF_encoded_df.to_csv(encoded_file, sep='\t', index=True)

    ## DNAacc
    print("[INFO] DNAacc AE running...")
    original_dim = DNAacc_train_X.shape[1]
    intermediate_dim = 256
    latent_dim = 64

    x = Input(shape=(original_dim,), name="mVAE_LR_DMAacc_Input")
    h = Dense(intermediate_dim, activation='relu', name="mVAE_LR_DMAacc_hidden")(x)
    z_mean = Dense(latent_dim, name="mVAE_LR_DMAacc_zmean")(h)
    z_log_sigma = Dense(latent_dim, name="mVAE_LR_DMAacc_zlogsigma")(h)

    z = Lambda(sampling, output_shape=(latent_dim,), name="mVAE_LR_DMAacc_z")([z_mean, z_log_sigma])

    h_decoded = Dense(intermediate_dim, activation='relu', name="mVAE_LR_DMAacc_hdecode")(z)
    x_decoded_mean = Dense(original_dim, activation='sigmoid', name="mVAE_LR_DMAacc_decodemean")(h_decoded)

    DNAacc_VAE = Model(x, x_decoded_mean)
    DNAacc_encoder = Model(x, z_mean)

    adam = optimizers.Adam(lr=learning_rate)
    DNAacc_VAE.compile(loss=vae_loss, optimizer=adam)

    output_model_file = os.path.join('', 'DNAacc_VAE_LR_architecture_' + st + '.png')
    plot_model(DNAacc_VAE, to_file=output_model_file)

    DNAacc_res = DNAacc_VAE.fit(DNAacc_train_X, DNAacc_train_X, epochs=epochs_AE,
                                batch_size=batch_size_AE,
                                validation_data=(DNAacc_val_X, DNAacc_val_X),
                                verbose=1)

    encoded_df = DNAacc_encoder.predict(DNAacc_train_X)
    DNAacc_encoded_df = pd.DataFrame(encoded_df, index=DNAacc_train_X.index)

    encoded_file = os.path.join('', 'encoded_DNAacc.txt')
    DNAacc_encoded_df.to_csv(encoded_file, sep='\t', index=True)

    # make the input dataframe
    frames = [Roadmap_encoded_df, TF_encoded_df, DNAacc_encoded_df, dataset_train_Y]
    dataset_input = pd.concat(frames, axis=1, sort=False)
    print(dataset_input.shape)

    X_train = dataset_input.drop('y_true', axis=1)
    y_train = dataset_input['y_true']
    print(X_train.shape)

    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X_train, y_train)
    #=============================================================================
    encoded_df = Roadmap_encoder.predict(Roadmap_test_X)
    Roadmap_encoded_df = pd.DataFrame(encoded_df, index=Roadmap_test_X.index)
    print(Roadmap_encoded_df.shape)

    encoded_df = TF_encoder.predict(TF_test_X)
    TF_encoded_df = pd.DataFrame(encoded_df, index=TF_test_X.index)
    print(TF_encoded_df.shape)

    encoded_df = DNAacc_encoder.predict(DNAacc_test_X)
    DNAacc_encoded_df = pd.DataFrame(encoded_df, index=DNAacc_test_X.index)
    print(DNAacc_encoded_df.shape)

    test_frames = [Roadmap_encoded_df, TF_encoded_df, DNAacc_encoded_df]
    test_X = pd.concat(test_frames, axis=1, sort=False)

    y_pred_x = model.predict_proba(test_X)[:, 1]
    y_pred = y_pred_x > 0.5

    accuracy_test = round(accuracy_score(np.array(dataset_test_Y, dtype=np.float32), y_pred),4)
    precision_test = round(precision_score(np.array(dataset_test_Y, dtype=np.float32), y_pred),4)
    recall_test = round(recall_score(np.array(dataset_test_Y, dtype=np.float32), y_pred),4)

    AUROC_test = roc_auc_score(np.array(dataset_test_Y, dtype=np.float32), y_pred_x)
    average_precision = round(average_precision_score(dataset_test_Y, y_pred_x), 4)

    fpr_roc, tpr_roc, thresholds_roc = roc_curve(dataset_test_Y, y_pred_x)
    precision_prc, recall_prc, thresholds_prc = precision_recall_curve(dataset_test_Y, y_pred_x)

    del Roadmap_encoder
    del Roadmap_VAE
    del TF_encoder
    del TF_VAE
    del DNAacc_encoder
    del DNAacc_VAE
    del model
    K.clear_session()

    return [accuracy_test, precision_test,recall_test, AUROC_test, average_precision, fpr_roc, tpr_roc,precision_prc, recall_prc]
