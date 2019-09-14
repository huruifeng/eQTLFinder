"""
Ruifeng Hu
09132019
UTHealth-SBMI
"""
import os
import datetime
import time

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S")

from models.DNN import DNN
from models.DNN_multiCategories import DNN_multi
from models.RF import RF
from models.RF_LR import RF_LR
from models.multiVAE_DNN import multiVAE_DNN
from models.multiVAE_Forest import multiVAE_Forest
from models.multiVAE_LR import multiVAE_LR
from models.LR import LR
from models.KNN import KNN

#==========================================================
tissue = 'Brain_Substantia_nigra'
print('[INFO] Loading '+tissue+' data...')

## load dataset
path = "E:\\RuifengHu\\Projects\\2018-11-05_eQTL\\Jupyter\\"
file = path + tissue+'_matrix_100.txt'
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
print(len(test_index))
print(len(val_index))
print(len(train_index))

dataset_test_X = dataset_input_X.loc[test_index,:]
dataset_val_X = dataset_input_X.loc[val_index,:]
dataset_train_X = dataset_input_X.loc[train_index, :]
print(dataset_test_X.shape)
print(dataset_val_X.shape)
print(dataset_train_X.shape)

print(y_true.shape)

dataset_test_Y = y_true.loc[test_index, :]
dataset_val_Y = y_true.loc[val_index,:]
dataset_train_Y = y_true.loc[train_index, :]

print("Train:", dataset_train_X.shape)
print("Val:", dataset_val_X.shape)
print("Test:", dataset_test_X.shape)

dataset_X = [dataset_train_X, dataset_val_X, dataset_test_X]
dataset_y = [dataset_train_Y, dataset_val_Y, dataset_test_Y]

# 0:accuracy, 1:precision, 2:recall,
# 3:AUROC, 4:average_precision
# 5:fpr_roc, 6:tpr_roc,
# 7:precision_prc, 8:recall_prc
print("**********DNN**********")
DNN_ls = DNN(dataset_X, dataset_y)
print("**********mDNN**********")
DNN_multi_ls = DNN_multi(dataset_X, dataset_y)
print("**********RF**********")
RF_ls = RF(dataset_X, dataset_y)
print("**********RF_LR**********")
RF_LR_ls = RF_LR(dataset_X, dataset_y)
print("**********mVAE_DNN**********")
multiVAE_DNN_ls = multiVAE_DNN(dataset_X, dataset_y)
print("**********mVAE_Forest**********")
multiVAE_Forest_ls = multiVAE_Forest(dataset_X, dataset_y)
print("**********mVAE_LR**********")
multiVAE_LR_ls = multiVAE_LR(dataset_X, dataset_y)
print("**********LR**********")
LR_ls = LR(dataset_X, dataset_y)
print("**********KNN**********")
KNN_ls = KNN(dataset_X, dataset_y)

print("DNN:",DNN_ls[0:5])
print("mDNN:", DNN_multi_ls[0:5])
print("RF:", RF_ls[0:5])
print("RF_LR:",RF_LR_ls[0:5])
print("mVAE_DNN:",multiVAE_DNN_ls[0:5])
print("mVAE_Forest:",multiVAE_Forest_ls[0:5])
print("mVAE_LR:",multiVAE_LR_ls[0:5])
print("LR:",LR_ls[0:5])
print("KNN:",KNN_ls[0:5])

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
label_str = "DNN-" + str(DNN_ls[3])
plt.plot(DNN_ls[5], DNN_ls[6], label=label_str)
label_str = "mDNN-" + str(DNN_multi_ls[3])
plt.plot(DNN_multi_ls[5], DNN_multi_ls[6], label=label_str)
label_str = "RF-" + str(RF_ls[3])
plt.plot(RF_ls[5], RF_ls[6], label=label_str)
label_str = "RF_LR-" + str(RF_LR_ls[3])
plt.plot(RF_LR_ls[5], RF_LR_ls[6], label=label_str)
label_str = "mVAE_DNN-" + str(multiVAE_DNN_ls[3])
plt.plot(multiVAE_DNN_ls[5], multiVAE_DNN_ls[6], label=label_str)
label_str = "mVAE_Forest-" + str(multiVAE_Forest_ls[3])
plt.plot(multiVAE_Forest_ls[5], multiVAE_Forest_ls[6], label=label_str)
label_str = "mVAE_LR-" + str(multiVAE_LR_ls[3])
plt.plot(multiVAE_LR_ls[5], multiVAE_LR_ls[6], label=label_str)
label_str = "LR-" + str(LR_ls[3])
plt.plot(LR_ls[5], LR_ls[6], label=label_str)
label_str = "KNN-" + str(KNN_ls[3])
plt.plot(KNN_ls[5], KNN_ls[6], label=label_str)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig('AUROC.pdf')
plt.show()

plt.figure(2)
plt.plot([0, 1], [1, 0], 'k--')
label_str = "DNN-" + str(DNN_ls[4])
plt.plot(DNN_ls[8], DNN_ls[7], label=label_str)
label_str = "mDNN-" + str(DNN_multi_ls[4])
plt.plot(DNN_multi_ls[8], DNN_multi_ls[7], label=label_str)
label_str = "RF-" + str(RF_ls[4])
plt.plot(RF_ls[8], RF_ls[7], label=label_str)
label_str = "RF_LR-" + str(RF_LR_ls[4])
plt.plot(RF_LR_ls[8], RF_LR_ls[7], label=label_str)
label_str = "mVAE_DNN-" + str(multiVAE_DNN_ls[4])
plt.plot(multiVAE_DNN_ls[8], multiVAE_DNN_ls[7], label=label_str)
label_str = "mVAE_Forest-" + str(multiVAE_Forest_ls[4])
plt.plot(multiVAE_Forest_ls[8], multiVAE_Forest_ls[7], label=label_str)
label_str = "mVAE_LR-" + str(multiVAE_LR_ls[4])
plt.plot(multiVAE_LR_ls[8], multiVAE_LR_ls[7], label=label_str)
label_str = "LR-" + str(LR_ls[4])
plt.plot(LR_ls[8], LR_ls[7], label=label_str)
label_str = "KNN-" + str(KNN_ls[4])
plt.plot(KNN_ls[8], KNN_ls[7], label=label_str)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig('P-R.pdf')
plt.show()






