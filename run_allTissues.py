"""
Ruifeng Hu
09-16-2019
UTHealth-SBMI
"""
import os
import subprocess
from multiprocessing import Pool


#=================================================
def runModels(tissue, file, result_folder):
    print(tissue)
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


if __name__ == '__main__':

    ##Path to the the files to be read
    path = "/collab2/CPH/rhu1/20190419_eQTL/DNN/V0_1v1"
    matrix_folder = path + "/Matrix_signal/full_matrix_rm_zeros100/"

    result_folder = "Results"
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    process_num = 30
    pool = Pool(processes=process_num)
    for tissue_i in os.listdir(matrix_folder):
        #print(tissue_i)
        #if 'Brain_Substantia_nigra' not in tissue_i:
        #    continue
        tissue =  tissue_i[:-11]
        matrix_file = matrix_folder + tissue_i

        pool.apply_async(runModels, args=(tissue, matrix_file, result_folder,))
    pool.close()
    pool.join()
    print('[INFO] Model run, Done !')

    print("**********plotResults**********")
    command_str = "pplotResult.py " + matrix_folder + " " + result_folder
    # os.system(command_str)
    subprocess.call(command_str, shell=True)
