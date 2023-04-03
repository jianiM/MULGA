# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 15:42:05 2022

@author: Jiani Ma
"""

import os 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import KFold
from numpy import linalg as LA
import matplotlib.pyplot as plt

def read_data(root_path,dataset,drug_sim_file,target_sim_file,dti_file,drug_encoder_path,target_encoder_path):
    drug_sim_path = os.path.join(root_path,dataset,drug_sim_file)
    target_sim_path = os.path.join(root_path,dataset,target_sim_file)
    dti_path = os.path.join(root_path,dataset,dti_file)
    SR = pd.read_excel(drug_sim_path,header=None).values
    A_orig = pd.read_excel(dti_path,header=None).values 
    A_orig_arr = A_orig.flatten()
    known_sample = np.nonzero(A_orig_arr)[0]       
    drug_encoder_list = pd.read_excel(drug_encoder_path,header=None).values
    target_encoder_list = pd.read_excel(target_encoder_path,header=None).values  
    drug_num = A_orig.shape[0]
    target_num = A_orig.shape[1]    
    return SR,A_orig,A_orig_arr,known_sample,drug_encoder_list,target_encoder_list,drug_num,target_num


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