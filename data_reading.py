# -*- coding: utf-8 -*-

"""
Created on Sun May 29 17:55:57 2022
@author: Jiani Ma
"""
import os 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import KFold
from numpy import linalg as LA
import matplotlib.pyplot as plt

def read_data(data_folder,drug_sim_path,target_sim_path,DTI_path):
    SR = pd.read_excel(os.path.join(data_folder, drug_sim_path),header=None).values
    SD = pd.read_excel(os.path.join(data_folder, target_sim_path),header=None).values
    A_orig = pd.read_excel(os.path.join(data_folder, DTI_path),header=None).values 
    A_orig_arr = A_orig.flatten()
    known_sample = np.nonzero(A_orig_arr)[0]   
    return SR,SD,A_orig,A_orig_arr,known_sample


def get_drug_dissimmat(drug_affinity_matrix,topk):
    drug_num  = drug_affinity_matrix.shape[0]
    drug_dissim_mat = np.zeros((drug_num,topk))
    index_list = np.arange(drug_num)
    for i in range(drug_num):
        score = drug_affinity_matrix[i]
        index_score = list(zip(index_list,score))
        sorted_result = sorted(index_score,key=lambda x: x[1],reverse=False)[1:topk+1]
        drug_id_list = np.zeros(topk)
        for j in range(topk):
            drug_id_list[j] = sorted_result[j][0]            
        drug_dissim_mat[i] = drug_id_list
    return drug_dissim_mat 
       


    






