#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 11:12:54 2022

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
import argparse
from numpy.linalg import norm
from random import normalvariate
from sklearn.model_selection import KFold
from numpy import linalg as LA

def compute_GSTtol(lam,p):
    pow1 = 1 / (2-p)
    pow2 = (p-1) / (2-p)
    root = 2 * lam *(1-p)
    gsttol = np.power(root,pow1) + lam*p *np.power(root,pow2)
    return gsttol

def GMST(A,lam,p):
    U,s,VT = LA.svd(A,full_matrices=False)
    gst_tol = compute_GSTtol(lam,p)
    gst_sigma_arr = np.zeros(len(s))
    for i in range(len(s)):
        sigma = s[i]        
        if sigma>gst_tol:
            gst_sigma = sigma 
            for k in range(2):
                gst_sigma = sigma - lam * p *np.power(gst_sigma,p-1)
        else: 
            gst_sigma = 0                 

        gst_sigma_arr[i] = gst_sigma  
    S = np.diag(gst_sigma_arr)
    gmst_tmp = np.dot(U,S)
    gmst = np.dot(gmst_tmp,VT)
    return gmst

def LaplacianMatrix(S): 
    m = S.shape[0]
    D = np.zeros((m,m))
    d = np.sum(S,axis=1)
    D = np.diag(d)
    snL = D - S
    return snL 

def update_W(X,U,mu_1,p):
    lam = 1/mu_1
    mat = X + lam*U 
    W = GMST(mat,lam,p)
    return W 

def update_X(lambda_d,lambda_t,Ld,Lt,mu_1,mu_2,W,Z,U,V,m,n):
    a = 2 * lambda_d * Ld + mu_1 * np.identity(m)
    b = 2 * lambda_t * Lt + mu_2 * np.identity(n)
    q = mu_1 * W + mu_2 * Z - U -V 
    X = linalg.solve_sylvester(a, b, q)    
    return X 

def update_Z(alpha,mu_2,A,V,X,A_mask):
    coff_one = alpha/mu_2
    coff_two = 1/mu_2
    coff_three = alpha/(alpha+mu_2)
    common_term = coff_one * np.multiply(A_mask,A) + coff_two * V + X 
    Z = common_term - coff_three*np.multiply(A_mask,common_term) 
    return Z 

def update_U_V(lastU,lastV,beta,X,W,Z):
    U = lastU + beta*(X-W)
    V = lastV + beta*(X-Z)
    return U, V 

def converge(X,W,Z): 
    XW_threshold = LA.norm(X-W)
    XZ_threshold = LA.norm(X-Z)
    return XW_threshold, XZ_threshold    

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

def get_negative_samples(mask,drug_dissimmat):    
    pos_num = np.sum(mask)     # 193
    pos_id = np.where(mask==1) # 2D postion of mask,2 * 193
    drug_id = pos_id[0]        # 193
    t_id = pos_id[1]           # 193 
    neg_mask = np.zeros_like(mask)  
    for i in range(pos_num):   # for each positive sample  
        d = drug_id[i]
        t = t_id[i] 
        pos_drug = drug_dissimmat[d]  # 10 
        for j in range(len(pos_drug)):
            neg_mask[pos_drug[j]][t] = 1 
    return neg_mask 
    
def get_metric(real_score, predict_score):
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num*np.arange(1, 1000)/1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]
    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1)-TP
    FN = real_score.sum()-TP
    TN = len(real_score.T)-TP-FP-FN
    fpr = FP/(FP+TN)
    tpr = TP/(TP+FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5*(x_ROC[1:]-x_ROC[:-1]).T*(y_ROC[:-1]+y_ROC[1:])
    recall_list = tpr
    precision_list = TP/(TP+FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5*(x_PR[1:]-x_PR[:-1]).T*(y_PR[:-1]+y_PR[1:])
    f1_score_list = 2*TP/(len(real_score.T)+TP-TN)
    accuracy_list = (TP+TN)/len(real_score.T)
    specificity_list = TN/(TN+FP)
    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return TP,FP,FN,TN,fpr,tpr,auc[0, 0], aupr[0, 0],f1_score, accuracy, recall, specificity, precision