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


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S")

#==========================================================
## fix random seed for reproducibility
np.random.seed(12)

#==========================================================
tissue = sys.argv[1]
# tissue = "Brain_Substantia_nigra"
Results = sys.argv[2]

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

model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)
#=============================================================================

y_pred_x = model.predict_proba(X_test)[:, 1]
y_pred = y_pred_x > 0.5

accuracy_test = accuracy_score(np.array(y_test, dtype=np.float32), y_pred)
precision_test = precision_score(np.array(y_test, dtype=np.float32), y_pred)
recall_test = recall_score(np.array(y_test, dtype=np.float32), y_pred)

AUROC_test = roc_auc_score(np.array(y_test, dtype=np.float32), y_pred_x)
average_precision = average_precision_score(y_test, y_pred_x)

fpr_roc, tpr_roc, thresholds_roc = roc_curve(y_test, y_pred_x)
precision_prc, recall_prc, thresholds_prc = precision_recall_curve(y_test, y_pred_x)
data_x = { "accuracy": np.array([accuracy_test, precision_test, recall_test, AUROC_test,average_precision]),
           "fpr": fpr_roc,
           "tpr": tpr_roc,
           "precision":precision_prc,
           "recall":recall_prc}
np.savez(Results+"/"+tissue+'-mVAE_LR.npz', **data_x)

#return [accuracy_test, precision_test,recall_test, AUROC_test, average_precision, fpr_roc, tpr_roc,precision_prc, recall_prc]
