"""
Ruifeng Hu
09132019
UTHealth-SBMI
"""


import datetime
import time

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S")

#==========================================================
## fix random seed for reproducibility
np.random.seed(12)

#==========================================================


def RF_LR(dataset_X,dataset_Y):
    dataset_train_X = dataset_X[0]
    dataset_val_X = dataset_X[1]
    dataset_test_X = dataset_X[2]

    dataset_train_Y = np.array(dataset_Y[0])
    dataset_val_Y = np.array(dataset_Y[1])
    dataset_test_Y = np.array(dataset_Y[2])

    X_train, X_train_lr, y_train, y_train_lr = train_test_split(dataset_train_X, dataset_train_Y, test_size=0.5)

    # Supervised transformation based on random forests
    rf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=2, random_state=0)
    rf_enc = OneHotEncoder(categories='auto')
    rf_lm = LogisticRegression(solver='lbfgs', max_iter=1000)
    rf.fit(X_train, np.ravel(y_train))
    rf_enc.fit(rf.apply(X_train))
    rf_lm.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)

    y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(dataset_test_X)))[:, 1]
    fpr_rf_lm, tpr_rf_lm, _ = roc_curve(dataset_test_Y, y_pred_rf_lm)
    y_pred_x = y_pred_rf_lm
    y_pred = y_pred_x > 0.5

    accuracy_test = round(accuracy_score(np.array(dataset_test_Y, dtype=np.float32), y_pred), 4)
    precision_test = round(precision_score(np.array(dataset_test_Y, dtype=np.float32), y_pred), 4)
    recall_test = round(recall_score(np.array(dataset_test_Y, dtype=np.float32), y_pred), 4)

    AUROC_test = roc_auc_score(np.array(dataset_test_Y, dtype=np.float32), y_pred_x)
    average_precision = round(average_precision_score(dataset_test_Y, y_pred_x), 4)

    fpr_roc, tpr_roc, thresholds_roc = roc_curve(dataset_test_Y, y_pred_x)
    precision_prc, recall_prc, thresholds_prc = precision_recall_curve(dataset_test_Y, y_pred_x)


    del rf
    del rf_enc
    del rf_lm


    return [accuracy_test, precision_test,recall_test, AUROC_test, average_precision, fpr_roc, tpr_roc,precision_prc, recall_prc]

