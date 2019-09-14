"""
Ruifeng Hu
09132019
UTHealth-SBMI
"""

import datetime
import time

import numpy as np

from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S")

#==========================================================
## fix random seed for reproducibility
np.random.seed(12)

#==========================================================


def KNN(dataset_X,dataset_Y):
    dataset_train_X = dataset_X[0]
    dataset_val_X = dataset_X[1]
    dataset_test_X = dataset_X[2]

    dataset_train_Y = dataset_Y[0]
    dataset_val_Y = dataset_Y[1]
    dataset_test_Y = dataset_Y[2]

    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=100)

    model.fit(dataset_train_X, dataset_train_Y)

    y_pred_x = model.predict_proba(dataset_test_X)[:, 1]
    y_pred = y_pred_x > 0.5

    accuracy_test = round(accuracy_score(np.array(dataset_test_Y, dtype=np.float32), y_pred), 4)
    precision_test = round(precision_score(np.array(dataset_test_Y, dtype=np.float32), y_pred), 4)
    recall_test = round(recall_score(np.array(dataset_test_Y, dtype=np.float32), y_pred), 4)

    AUROC_test = roc_auc_score(np.array(dataset_test_Y, dtype=np.float32), y_pred_x)
    average_precision = round(average_precision_score(dataset_test_Y, y_pred_x), 4)

    fpr_roc, tpr_roc, thresholds_roc = roc_curve(dataset_test_Y, y_pred_x)
    precision_prc, recall_prc, thresholds_prc = precision_recall_curve(dataset_test_Y, y_pred_x)

    del model

    return [accuracy_test, precision_test,recall_test, AUROC_test, average_precision, fpr_roc, tpr_roc,precision_prc, recall_prc]

