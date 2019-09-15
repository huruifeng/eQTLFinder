"""
Ruifeng Hu
09132019
UTHealth-SBMI
"""

import sys
import datetime
import time

import numpy as np
import pandas as pd

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
#====================================================================
dataset_train_X = dataset_X[0]
dataset_val_X = dataset_X[1]
dataset_test_X = dataset_X[2]

dataset_train_Y = dataset_Y[0]
dataset_val_Y = dataset_Y[1]
dataset_test_Y = dataset_Y[2]

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=100)

model.fit(dataset_train_X, np.ravel(dataset_train_Y))

y_pred_x = model.predict_proba(dataset_test_X)[:, 1]
y_pred = y_pred_x > 0.5

accuracy_test = accuracy_score(np.array(dataset_test_Y, dtype=np.float32), y_pred)
precision_test = precision_score(np.array(dataset_test_Y, dtype=np.float32), y_pred)
recall_test = recall_score(np.array(dataset_test_Y, dtype=np.float32), y_pred)

AUROC_test = roc_auc_score(np.array(dataset_test_Y, dtype=np.float32), y_pred_x)
average_precision = average_precision_score(dataset_test_Y, y_pred_x)

fpr_roc, tpr_roc, thresholds_roc = roc_curve(dataset_test_Y, y_pred_x)
precision_prc, recall_prc, thresholds_prc = precision_recall_curve(dataset_test_Y, y_pred_x)

del model

data_x = { "accuracy": np.array([accuracy_test, precision_test, recall_test, AUROC_test,average_precision]),
           "fpr": fpr_roc,
           "tpr": tpr_roc,
           "precision":precision_prc,
           "recall":recall_prc}
np.savez("Results/"+tissue+'-KNN.npz', **data_x)

#return [accuracy_test, precision_test,recall_test, AUROC_test, average_precision, fpr_roc, tpr_roc,precision_prc, recall_prc]

