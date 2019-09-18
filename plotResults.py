"""
Ruifeng Hu
09-16-2019
UTHealth-SBMI
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

def plotRes(matrix_folder,result_folder):
    # ==============================================================================
    # The PDF document of AUROC
    with PdfPages(result_folder + "/AUROC.pdf") as auroc_pdf, PdfPages(result_folder + "/P-R.pdf") as pr_pdf:
        for tissue_i in os.listdir(matrix_folder):
            tissue = tissue_i[:-11]
            # get the results
            data_DNN = np.load(result_folder + "/" + tissue + "-DNN.npz", allow_pickle=True)
            data_mDNN = np.load(result_folder + "/" + tissue + "-mDNN.npz", allow_pickle=True)
            data_RF = np.load(result_folder + "/" + tissue + "-RF.npz", allow_pickle=True)
            data_LR = np.load(result_folder + "/" + tissue + "-LR.npz", allow_pickle=True)
            data_RF_LR = np.load(result_folder + "/" + tissue + "-RF_LR.npz", allow_pickle=True)
            data_KNN = np.load(result_folder + "/" + tissue + "-KNN.npz", allow_pickle=True)
            data_mVAE_DNN = np.load(result_folder + "/" + tissue + "-mVAE_DNN.npz", allow_pickle=True)
            data_mVAE_Forest = np.load(result_folder + "/" + tissue + "-mVAE_Forest.npz", allow_pickle=True)
            data_mVAE_LR = np.load(result_folder + "/" + tissue + "-mVAE_LR.npz", allow_pickle=True)

            # print the result
            # accuracy: [0:accuracy, 1:precision, 2:recall, 3:AUROC, 4:average_precision]
            # fpr: fpr_roc, tpr: tpr_roc,
            # precision: precision_prc, recall :recall_prc
            print("DNN:", data_DNN["accuracy"])
            print("mDNN:", data_mDNN["accuracy"])
            print("RF:", data_RF["accuracy"])
            print("LR:", data_LR["accuracy"])
            print("RF_LR:", data_RF_LR["accuracy"])
            print("KNN:", data_KNN["accuracy"])
            print("mVAE_DNN:", data_mVAE_DNN["accuracy"])
            print("mVAE_Forest:", data_mVAE_Forest["accuracy"])
            print("mVAE_LR:", data_mVAE_LR["accuracy"])

            # Create a figure instance (ie. a new page)
            fig = plt.figure()
            # Plot whatever you wish to plot
            plt.figure(1)
            plt.plot([0, 1], [0, 1], "k--")
            label_str = "DNN-" + "{0:.2f}".format(data_DNN["accuracy"][3] * 100) + "%"
            plt.plot(data_DNN["fpr"], data_DNN["tpr"], label=label_str)

            label_str = "mDNN-" + "{0:.2f}".format(data_mDNN["accuracy"][3] * 100) + "%"
            plt.plot(data_mDNN["fpr"], data_mDNN["tpr"], label=label_str)

            label_str = "RF-" + "{0:.2f}".format(data_RF["accuracy"][3] * 100) + "%"
            plt.plot(data_RF["fpr"], data_RF["tpr"], label=label_str)

            label_str = "LR-" + "{0:.2f}".format(data_LR["accuracy"][3] * 100) + "%"
            plt.plot(data_LR["fpr"], data_LR["tpr"], label=label_str)

            label_str = "RF_LR-" + "{0:.2f}".format(data_RF_LR["accuracy"][3] * 100) + "%"
            plt.plot(data_RF_LR["fpr"], data_RF_LR["tpr"], label=label_str)

            label_str = "KNN-" + "{0:.2f}".format(data_KNN["accuracy"][3] * 100) + "%"
            plt.plot(data_KNN["fpr"], data_KNN["tpr"], label=label_str)

            label_str = "mVAE_DNN-" + "{0:.2f}".format(data_mVAE_DNN["accuracy"][3] * 100) + "%"
            plt.plot(data_mVAE_DNN["fpr"], data_mVAE_DNN["tpr"], label=label_str)

            label_str = "mVAE_Forest-" + "{0:.2f}".format(data_mVAE_Forest["accuracy"][3] * 100) + "%"
            plt.plot(data_mVAE_Forest["fpr"], data_mVAE_Forest["tpr"], label=label_str)

            label_str = "mVAE_LR-" + "{0:.2f}".format(data_mVAE_LR["accuracy"][3] * 100) + "%"
            plt.plot(data_mVAE_LR["fpr"], data_mVAE_LR["tpr"], label=label_str)
            plt.xlabel("False positive rate")
            plt.ylabel("True positive rate")
            plt.title('ROC curve - ' + tissue)
            plt.legend(loc="best", fontsize=6)

            # Done with the page
            auroc_pdf.savefig(fig)

            ##=========================================
            plt.figure()
            plt.plot([0, 1], [1, 0], "k--")
            label_str = "DNN-" + "{0:.2f}".format(data_DNN["accuracy"][4] * 100) + "%"
            plt.plot(data_DNN["recall"], data_DNN["precision"], label=label_str)

            label_str = "mDNN-" + "{0:.2f}".format(data_mDNN["accuracy"][4] * 100) + "%"
            plt.plot(data_mDNN["recall"], data_mDNN["precision"], label=label_str)

            label_str = "RF-" + "{0:.2f}".format(data_RF["accuracy"][4] * 100) + "%"
            plt.plot(data_RF["recall"], data_RF["precision"], label=label_str)

            label_str = "LR-" + "{0:.2f}".format(data_LR["accuracy"][4] * 100) + "%"
            plt.plot(data_LR["recall"], data_LR["precision"], label=label_str)

            label_str = "RF_LR-" + "{0:.2f}".format(data_RF_LR["accuracy"][4] * 100) + "%"
            plt.plot(data_RF_LR["recall"], data_RF_LR["precision"], label=label_str)

            label_str = "KNN-" + "{0:.2f}".format(data_KNN["accuracy"][4] * 100) + "%"
            plt.plot(data_KNN["recall"], data_KNN["precision"], label=label_str)

            label_str = "mVAE_DNN-" + "{0:.2f}".format(data_mVAE_DNN["accuracy"][4] * 100) + "%"
            plt.plot(data_mVAE_DNN["recall"], data_mVAE_DNN["precision"], label=label_str)

            label_str = "mVAE_Forest-" + "{0:.2f}".format(data_mVAE_Forest["accuracy"][4] * 100) + "%"
            plt.plot(data_mVAE_Forest["recall"], data_mVAE_Forest["precision"], label=label_str)

            label_str = "mVAE_LR-" + "{0:.2f}".format(data_mVAE_LR["accuracy"][4] * 100) + "%"
            plt.plot(data_mVAE_LR["recall"], data_mVAE_LR["precision"], label=label_str)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title('P-R curve - ' + tissue)
            plt.legend(loc="best", fontsize=6)
            # Done with the page
            pr_pdf.savefig(fig)

            data_DNN.close()
            data_mDNN.close()
            data_RF.close()
            data_RF_LR.close()
            data_LR.close()
            data_KNN.close()
            data_mVAE_DNN.close()
            data_mVAE_Forest.close()
            data_mVAE_LR.close()

if __name__ == '__main__':

    ##Path to the the files to be read
    path = "/collab2/CPH/rhu1/20190419_eQTL/DNN/V0_1v1"
    matrix_folder = path + "/Matrix_signal/full_matrix_rm_zeros100/"

    result_folder = "Results"

    print("**********plotResults**********")
    plotRes(matrix_folder,result_folder)
    print("**********Done**********")

