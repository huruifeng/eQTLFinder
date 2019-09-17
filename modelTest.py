"""
Ruifeng Hu
09132019
UTHealth-SBMI
"""
import os
import subprocess

import numpy as np
import matplotlib.pyplot as plt


tissue = "Brain_Substantia_nigra"
#path = "C:\\Users\\rhu1\\Dropbox\\Temp\\"
path = "C:\\Users\\hurui\\Dropbox\\Temp\\"
file = path + tissue+'_matrix_1v3_hit.txt'
result_folder = "Results"

if not os.path.exists(result_folder):
    os.mkdir(result_folder)

print("**********DNN**********")
command_str = "python3 DNN.py "+ tissue + " " + file + " "+result_folder
#os.system(command_str)
subprocess.call(command_str, shell=True)

print("**********mDNN**********")
epoach_x = 12
command_str = "python3 mDNN.py "+ tissue+ " " + file +  " " +result_folder
#os.system(command_str)
subprocess.call(command_str, shell=True)

print("**********RF**********")
command_str = "python3 RF.py "+ tissue+ " " + file + " "+result_folder
#os.system(command_str)
subprocess.call(command_str, shell=True)

print("**********LR**********")
command_str = "python3 LR.py "+ tissue+ " " + file + " "+result_folder
#os.system(command_str)
subprocess.call(command_str, shell=True)

print("**********RF_LR**********")
command_str = "python3 RF_LR.py "+ tissue+ " " + file + " "+result_folder
#os.system(command_str)
subprocess.call(command_str, shell=True)

print("**********KNN**********")
command_str = "python3 KNN.py "+ tissue+ " " + file + " "+result_folder
#os.system(command_str)
subprocess.call(command_str, shell=True)

command_str = "python3 VAE.py "+ tissue+ " " + file + " "+result_folder
#os.system(command_str)
subprocess.call(command_str, shell=True)
print("**********mVAE_DNN**********")
command_str = "python3 mVAE_DNN.py "+ tissue + " "+result_folder
#os.system(command_str)
subprocess.call(command_str, shell=True)

print("**********mVAE_Forest**********")
command_str = "python3 mVAE_Forest.py "+ tissue + " "+result_folder
#os.system(command_str)
subprocess.call(command_str, shell=True)

print("**********mVAE_LR**********")
command_str = "python3 mVAE_LR.py "+ tissue + " "+result_folder
#os.system(command_str)
subprocess.call(command_str, shell=True)


#===================================================================
data_DNN = np.load(result_folder+"/"+tissue+"-DNN.npz", allow_pickle=True)
data_mDNN = np.load(result_folder+"/"+tissue+"-mDNN.npz", allow_pickle=True)
data_RF = np.load(result_folder+"/"+tissue+"-RF.npz", allow_pickle=True)
data_LR = np.load(result_folder+"/"+tissue+"-LR.npz", allow_pickle=True)
data_RF_LR = np.load(result_folder+"/"+tissue+"-RF_LR.npz", allow_pickle=True)
data_KNN = np.load(result_folder+"/"+tissue+"-KNN.npz", allow_pickle=True)
data_mVAE_DNN = np.load(result_folder+"/"+tissue+"-mVAE_DNN.npz", allow_pickle=True)
data_mVAE_Forest = np.load(result_folder+"/"+tissue+"-mVAE_Forest.npz", allow_pickle=True)
data_mVAE_LR = np.load(result_folder+"/"+tissue+"-mVAE_LR.npz", allow_pickle=True)

# accuracy: [0:accuracy, 1:precision, 2:recall, 3:AUROC, 4:average_precision]
# fpr: fpr_roc,
# tpr: tpr_roc,
# precision: precision_prc,
# recall :recall_prc

print("DNN:",data_DNN["accuracy"])
print("mDNN:", data_mDNN["accuracy"])
print("RF:", data_RF["accuracy"])
print("LR:",data_LR["accuracy"])
print("RF_LR:",data_RF_LR["accuracy"])
print("KNN:",data_KNN["accuracy"])
print("mVAE_DNN:",data_mVAE_DNN["accuracy"])
print("mVAE_Forest:",data_mVAE_Forest["accuracy"])
print("mVAE_LR:",data_mVAE_LR["accuracy"])
#
plt.figure(1)
plt.plot([0, 1], [0, 1], "k--")
label_str = "DNN-" + "{0:.2f}".format(data_DNN["accuracy"][3]*100) +"%"
plt.plot(data_DNN["fpr"], data_DNN["tpr"], label=label_str)

label_str = "mDNN-" + "{0:.2f}".format(data_mDNN["accuracy"][3]*100) +"%"
plt.plot(data_mDNN["fpr"], data_mDNN["tpr"], label=label_str)

label_str = "RF-" + "{0:.2f}".format(data_RF["accuracy"][3]*100) +"%"
plt.plot(data_RF["fpr"], data_RF["tpr"], label=label_str)

label_str = "LR-" + "{0:.2f}".format(data_LR["accuracy"][3]*100) +"%"
plt.plot(data_LR["fpr"], data_LR["tpr"], label=label_str)

label_str = "RF_LR-" + "{0:.2f}".format(data_RF_LR["accuracy"][3]*100) +"%"
plt.plot(data_RF_LR["fpr"], data_RF_LR["tpr"], label=label_str)

label_str = "KNN-" + "{0:.2f}".format(data_KNN["accuracy"][3]*100) +"%"
plt.plot(data_KNN["fpr"], data_KNN["tpr"], label=label_str)

label_str = "mVAE_DNN-" + "{0:.2f}".format(data_mVAE_DNN["accuracy"][3]*100) +"%"
plt.plot(data_mVAE_DNN["fpr"], data_mVAE_DNN["tpr"], label=label_str)

label_str = "mVAE_Forest-" + "{0:.2f}".format(data_mVAE_Forest["accuracy"][3]*100) +"%"
plt.plot(data_mVAE_Forest["fpr"], data_mVAE_Forest["tpr"], label=label_str)

label_str = "mVAE_LR-" + "{0:.2f}".format(data_mVAE_LR["accuracy"][3]*100) +"%"
plt.plot(data_mVAE_LR["fpr"], data_mVAE_LR["tpr"], label=label_str)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curve")
plt.legend(loc="best", fontsize=6)
plt.savefig(result_folder+"/"+tissue+"_AUROC.pdf")

##=========================================
plt.figure(2)
plt.plot([0, 1], [1, 0], "k--")
label_str = "DNN-" + "{0:.2f}".format(data_DNN["accuracy"][4]*100) +"%"
plt.plot(data_DNN["recall"], data_DNN["precision"], label=label_str)

label_str = "mDNN-" + "{0:.2f}".format(data_mDNN["accuracy"][4]*100) +"%"
plt.plot(data_mDNN["recall"], data_mDNN["precision"], label=label_str)

label_str = "RF-" + "{0:.2f}".format(data_RF["accuracy"][4]*100) +"%"
plt.plot(data_RF["recall"], data_RF["precision"], label=label_str)

label_str = "LR-" + "{0:.2f}".format(data_LR["accuracy"][4]*100) +"%"
plt.plot(data_LR["recall"], data_LR["precision"], label=label_str)

label_str = "RF_LR-" + "{0:.2f}".format(data_RF_LR["accuracy"][4]*100) +"%"
plt.plot(data_RF_LR["recall"], data_RF_LR["precision"], label=label_str)

label_str = "KNN-" + "{0:.2f}".format(data_KNN["accuracy"][4]*100) +"%"
plt.plot(data_KNN["recall"], data_KNN["precision"], label=label_str)

label_str = "mVAE_DNN-" + "{0:.2f}".format(data_mVAE_DNN["accuracy"][4]*100) +"%"
plt.plot(data_mVAE_DNN["recall"], data_mVAE_DNN["precision"], label=label_str)

label_str = "mVAE_Forest-" + "{0:.2f}".format(data_mVAE_Forest["accuracy"][4]*100) +"%"
plt.plot(data_mVAE_Forest["recall"], data_mVAE_Forest["precision"], label=label_str)

label_str = "mVAE_LR-" + "{0:.2f}".format(data_mVAE_LR["accuracy"][4]*100) +"%"
plt.plot(data_mVAE_LR["recall"], data_mVAE_LR["precision"], label=label_str)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("ROC curve")
plt.legend(loc="best",fontsize=6)
plt.savefig(result_folder+"/"+tissue+"_P-R.pdf")

data_DNN.close()
data_mDNN.close()
data_RF.close()
data_RF_LR.close()
data_LR.close()
data_KNN.close()
data_mVAE_DNN.close()
data_mVAE_Forest.close()
data_mVAE_LR.close()




