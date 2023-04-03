#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 11:10:54 2022
@author: Jiani Ma
"""
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import matplotlib as mpl
#mpl.use('Agg')
from scipy import linalg
import matplotlib.pyplot as plt
from math import sqrt  
import argparse
from numpy.linalg import norm
from sklearn.model_selection import KFold
from numpy import linalg as LA
from utils import * 
from data_reading import * 
np.random.seed(1)  

def parse_args():    
    parser = argparse.ArgumentParser(description="This is a template of machine learning developping source code.")
    parser.add_argument('-data_folder', '--data_folder_topofallfeature', type=str, default="/home/amber/MULGA/LRSpNM/data/KIBA/") 
    parser.add_argument('-drug_sim_path', '--drug_sim_path_topofallfeature', type=str,  default="drug_sim_mat.xlsx") 
    parser.add_argument('-target_sim_path', '--target_sim_path_topofallfeature', type=str, default="target_sim_mat.xlsx") 
    parser.add_argument('-DTI_path', '--DTI_path_topofallfeature', type=str, default="dti_mat.xlsx") 
    parser.add_argument('-m', '--m_topofallfeature', type=int, default=1720,help='drug number') 
    parser.add_argument('-n', '--n_topofallfeature', type=int, default=220,help='target number') 
    parser.add_argument('-lambda_d', '--lambda_d_topofallfeature', type=float, default=2,help='Performing k-fold for cross validation.') 
    parser.add_argument('-lambda_t', '--lambda_t_topofallfeature', type=float, default=2,help='Performing k-fold for cross validation.') 
    parser.add_argument('-alpha', '--alpha_topofallfeature', type=float, default=1,help='Performing k-fold for cross validation.') 
    parser.add_argument('-mu_1', '--mu_1_topofallfeature', type=float, default=1,help='Performing k-fold for cross validation.') 
    parser.add_argument('-topk', '--topk_topofallfeature', type=int, default=1)     
    parser.add_argument('-p', '--p_topofallfeature', type=float, default=0.8,help='determine the schatten p norm') 
    parser.add_argument('-n_splits', '--n_splits_topofallfeature', type=int, default=10)     
    parser.add_argument('-epoch_num', '--epoch_num_topofallfeature', type=int, default=100)     
    return parser.parse_args() 

if __name__=="__main__":
    args = parse_args()
    data_folder = args.data_folder_topofallfeature
    drug_sim_path = args.drug_sim_path_topofallfeature
    target_sim_path = args.target_sim_path_topofallfeature
    DTI_path = args.DTI_path_topofallfeature
    m = args.m_topofallfeature
    n = args.n_topofallfeature
    lambda_d = args.lambda_d_topofallfeature    #(2_2, 2-1,20,21,22)
    lambda_t = args.lambda_t_topofallfeature    #(2_2, 2-1,20,21,22)
    alpha = args.alpha_topofallfeature          #(2_2, 2-1,20,21,22)
    mu_1 = args.mu_1_topofallfeature            #(2_2, 2-1,20,21,22)
    topk = args.topk_topofallfeature 
    beta = mu_2 = mu_1
    p = args.p_topofallfeature
    n_splits = args.n_splits_topofallfeature
    epoch_num = args.epoch_num_topofallfeature

    SR,SD,A_orig,A_orig_arr,known_sample = read_data(data_folder,drug_sim_path,target_sim_path,DTI_path)     
    Ld = LaplacianMatrix(SR)  # drug 
    Lt = LaplacianMatrix(SD)  # target
    drug_dissimmat = get_drug_dissimmat(SR,topk = topk).astype(int)
    kf = KFold(n_splits, shuffle=True)      #10 fold
    train_all=[]
    test_all=[]
    for train_ind,test_ind in kf.split(known_sample):  
        train_all.append(train_ind) 
        test_all.append(test_ind)
    
    overall_auroc = 0 
    overall_aupr = 0
    overall_f1 = 0 
    overall_acc = 0
    overall_recall = 0    
    overall_specificity = 0
    overall_precision = 0 
    A_unknown_mask = 1 - A_orig  

    for fold_int in range(n_splits):
        print("fold_int:",fold_int)
        pos_train_id = train_all[fold_int]
        pos_train = known_sample[pos_train_id]
        pos_train_mask_list = np.zeros_like(A_orig_arr)
        pos_train_mask_list[pos_train] = 1 
        pos_train_mask = pos_train_mask_list.reshape((m,n))
        neg_train_mask_candidate = get_negative_samples(pos_train_mask,drug_dissimmat)       
        neg_train_mask = np.multiply(neg_train_mask_candidate, A_unknown_mask)    
        A_train_mat = np.copy(pos_train_mask)
        A_train_mask = np.copy(pos_train_mask) #pos_train_mask + neg_train_mask 
        lastW = lastX = lastZ = lastU = lastV = np.zeros((m,n))
        for i in range(epoch_num):
            currentW = update_W(lastX,lastU,mu_1,p)
            currentZ = update_Z(alpha,mu_2,A_train_mat,lastV,lastX,A_train_mask)
            currentX = update_X(lambda_d,lambda_t,Ld,Lt,mu_1,mu_2,currentW,currentZ,lastU,lastV,m,n)
            currentU, currentV = update_U_V(lastU,lastV,beta,currentX,currentW,currentZ)
            errorXW, errorXZ = converge(currentX,currentW,currentZ)
            print("error_XW:",errorXW)
            print("error_XZ:",errorXZ)
            lastW = currentW
            lastZ = currentZ 
            lastX = currentX 
            lastU = currentU 
            lastV = currentV                
        XSD_hat_arr = currentX.flatten()  # predicted results flatten         
        pos_test_id = test_all[fold_int]
        pos_test = known_sample[pos_test_id]
        pos_test_mask_list = np.zeros_like(A_orig_arr)
        pos_test_mask_list[pos_test] =1 
        pos_test_mask = pos_test_mask_list.reshape((m,n))
        neg_test_mask_candidate = get_negative_samples(pos_test_mask, drug_dissimmat)
        neg_test_mask = np.multiply(neg_test_mask_candidate, A_unknown_mask)
        neg_test = np.where(neg_test_mask.flatten() ==1)[0]                         
        pos_test_samples = XSD_hat_arr[pos_test]
        neg_test_samples = XSD_hat_arr[neg_test]       
        pos_labels = np.ones_like(pos_test_samples)
        neg_labels = np.zeros_like(neg_test_samples)        
        labels = np.hstack((pos_labels,neg_labels))
        scores = np.hstack((pos_test_samples,neg_test_samples))
        TP,FP,FN,TN,fpr,tpr,auroc,aupr,f1_score, accuracy, recall, specificity, precision = get_metric(labels,scores)
        print('TP:',TP)
        print('FP:',FP)
        print('FN:',FN)
        print('TN:',FN)
        print('fpr:',fpr)
        print('tpr:',tpr)
        print('auroc:',auroc)
        print('aupr:',aupr)
        print('f1_score:',f1_score)
        print('acc:',accuracy)
        print('recall:',recall)
        print('specificity:',specificity)
        print('precision:',precision)
        overall_auroc += auroc
        overall_aupr += aupr
        overall_f1 += f1_score
        overall_acc += accuracy
        overall_recall += recall
        overall_specificity +=specificity
        overall_precision += precision
    auroc_ = overall_auroc/n_splits
    aupr_ = overall_aupr/n_splits
    f1_ = overall_f1/n_splits
    acc_ = overall_acc/n_splits
    recall_ = overall_recall/n_splits
    specificity_ = overall_specificity/n_splits
    precision_ = overall_precision/n_splits
    print('mean_auroc:',auroc_)
    print('mean_aupr:',aupr_)
        
        
        
        
        
        
        
        
        
        
        

     
        
        
        
        
        
        
        
    
