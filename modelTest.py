"""
Ruifeng Hu
09132019
UTHealth-SBMI
"""
import os
import subprocess


# 0:accuracy, 1:precision, 2:recall,
# 3:AUROC, 4:average_precision
# 5:fpr_roc, 6:tpr_roc,
# 7:precision_prc, 8:recall_prc

tissue = "Brain_Substantia_nigra"

print("**********DNN**********")
command_str = 'python3 DNN.py '+ tissue
#os.system(command_str)
subprocess.call(command_str, shell=True)

print("**********mDNN**********")
command_str = 'python3 mDNN.py '+ tissue
#os.system(command_str)
subprocess.call(command_str, shell=True)

print("**********RF**********")
command_str = 'python3 RF.py '+ tissue
#os.system(command_str)
subprocess.call(command_str, shell=True)

print("**********LR**********")
command_str = 'python3 LR.py '+ tissue
#os.system(command_str)
subprocess.call(command_str, shell=True)

print("**********RF_LR**********")
command_str = 'python3 RF_LR.py '+ tissue
#os.system(command_str)
subprocess.call(command_str, shell=True)

print("**********KNN**********")
command_str = 'python3 KNN.py '+ tissue
#os.system(command_str)
subprocess.call(command_str, shell=True)

command_str = 'python3 VAE.py '+ tissue
#os.system(command_str)
subprocess.call(command_str, shell=True)
print("**********mVAE_DNN**********")
command_str = 'python3 mVAE_DNN.py '+ tissue
#os.system(command_str)
subprocess.call(command_str, shell=True)

print("**********mVAE_Forest**********")
command_str = 'python3 mVAE_Forest.py '+ tissue
#os.system(command_str)
subprocess.call(command_str, shell=True)

print("**********mVAE_LR**********")
command_str = 'python3 mVAE_LR.py '+ tissue
#os.system(command_str)
subprocess.call(command_str, shell=True)




#
# print("DNN:",DNN_ls[0:5])
# print("mDNN:", DNN_multi_ls[0:5])
# print("RF:", RF_ls[0:5])
# print("RF_LR:",RF_LR_ls[0:5])
# print("mVAE_DNN:",multiVAE_DNN_ls[0:5])
# print("mVAE_Forest:",multiVAE_Forest_ls[0:5])
# print("mVAE_LR:",multiVAE_LR_ls[0:5])
# print("LR:",LR_ls[0:5])
# print("KNN:",KNN_ls[0:5])
#
# plt.figure(1)
# plt.plot([0, 1], [0, 1], 'k--')
# label_str = "DNN-" + str(DNN_ls[3])
# plt.plot(DNN_ls[5], DNN_ls[6], label=label_str)
# label_str = "mDNN-" + str(DNN_multi_ls[3])
# plt.plot(DNN_multi_ls[5], DNN_multi_ls[6], label=label_str)
# label_str = "RF-" + str(RF_ls[3])
# plt.plot(RF_ls[5], RF_ls[6], label=label_str)
# label_str = "RF_LR-" + str(RF_LR_ls[3])
# plt.plot(RF_LR_ls[5], RF_LR_ls[6], label=label_str)
# label_str = "mVAE_DNN-" + str(multiVAE_DNN_ls[3])
# plt.plot(multiVAE_DNN_ls[5], multiVAE_DNN_ls[6], label=label_str)
# label_str = "mVAE_Forest-" + str(multiVAE_Forest_ls[3])
# plt.plot(multiVAE_Forest_ls[5], multiVAE_Forest_ls[6], label=label_str)
# label_str = "mVAE_LR-" + str(multiVAE_LR_ls[3])
# plt.plot(multiVAE_LR_ls[5], multiVAE_LR_ls[6], label=label_str)
# label_str = "LR-" + str(LR_ls[3])
# plt.plot(LR_ls[5], LR_ls[6], label=label_str)
# label_str = "KNN-" + str(KNN_ls[3])
# plt.plot(KNN_ls[5], KNN_ls[6], label=label_str)
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve')
# plt.legend(loc='best')
# plt.savefig('Results/AUROC.pdf')

#
# plt.figure(2)
# plt.plot([0, 1], [1, 0], 'k--')
# label_str = "DNN-" + str(DNN_ls[4])
# plt.plot(DNN_ls[8], DNN_ls[7], label=label_str)
# label_str = "mDNN-" + str(DNN_multi_ls[4])
# plt.plot(DNN_multi_ls[8], DNN_multi_ls[7], label=label_str)
# label_str = "RF-" + str(RF_ls[4])
# plt.plot(RF_ls[8], RF_ls[7], label=label_str)
# label_str = "RF_LR-" + str(RF_LR_ls[4])
# plt.plot(RF_LR_ls[8], RF_LR_ls[7], label=label_str)
# label_str = "mVAE_DNN-" + str(multiVAE_DNN_ls[4])
# plt.plot(multiVAE_DNN_ls[8], multiVAE_DNN_ls[7], label=label_str)
# label_str = "mVAE_Forest-" + str(multiVAE_Forest_ls[4])
# plt.plot(multiVAE_Forest_ls[8], multiVAE_Forest_ls[7], label=label_str)
# label_str = "mVAE_LR-" + str(multiVAE_LR_ls[4])
# plt.plot(multiVAE_LR_ls[8], multiVAE_LR_ls[7], label=label_str)
# label_str = "LR-" + str(LR_ls[4])
# plt.plot(LR_ls[8], LR_ls[7], label=label_str)
# label_str = "KNN-" + str(KNN_ls[4])
# plt.plot(KNN_ls[8], KNN_ls[7], label=label_str)
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('ROC curve')
# plt.legend(loc='best')
# plt.savefig('Results/P-R.pdf')






