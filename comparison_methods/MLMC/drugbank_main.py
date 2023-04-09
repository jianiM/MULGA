#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 23:10:55 2020
@author: Ma
"""

import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
from scipy import interp
import matplotlib.pyplot as plt
from math import sqrt  
from sklearn.metrics import roc_curve, auc
import argparse
from numpy.linalg import norm
from random import normalvariate
from math import sqrt
from numpy import linalg as LA
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc,average_precision_score,f1_score,accuracy_score

"""
Data process
"""

def read_data(data_folder,drug_sim_path,target_sim_path,DTI_path):
    XSS = pd.read_excel(os.path.join(data_folder, drug_sim_path),header=None).values
    XDD = pd.read_excel(os.path.join(data_folder, target_sim_path),header=None).values
    XSD = pd.read_excel(os.path.join(data_folder, DTI_path),header=None).values 
    XSD_mask_arr = XSD.flatten()
    known_samples = np.nonzero(XSD_mask_arr)[0]   
    return XSS,XDD,XSD,XSD_mask_arr,known_samples    
    

def Combine_X(XSD_train_matrix,XSS,XSD_train_mask,XDD):
    XSS_mask = np.ones_like(XSS)
    XDD_mask = np.ones_like(XDD)
    X_row1 = np.hstack((XSS,XSD_train_matrix))
    X_row2 = np.hstack((XSD_train_matrix.T,XDD))
    X = np.vstack((X_row1,X_row2))
    X_mask_row1 = np.hstack((XSS_mask,XSD_train_mask))
    X_mask_row2 = np.hstack((XSD_train_mask.T,XDD_mask))
    X_mask = np.vstack((X_mask_row1,X_mask_row2))
    return X, X_mask

def svt(A,tol):
    U,s,VT = LA.svd(A,full_matrices=False)
    shrink_s = s - tol
    shrink_s[shrink_s<0]=0
    S = np.diag(shrink_s)
    svtm_tmp = np.dot(U,S)
    svtm = np.dot(svtm_tmp,VT)
    return svtm

def D_update(mu,R,Y):
    tol = 1/mu
    D1 = R - tol*Y
    D = svt(D1,tol)   
    return D

def R_update(lamda,mu,X_mask,X,Y,D):
    tol = 1/mu
    coef1 =lamda/mu
    coef2 = lamda/(lamda+mu)
    R1=coef1*np.multiply(X_mask,X)+tol*Y+D
    R = R1-coef2*np.multiply(X_mask,R1)
    return R

def Y_update(lastY,delta,D,R):
    Y = lastY + delta*(D-R)
    return Y

def Converge_threshold(lastD,currentD,R,X_mask):
    D_error_mat = np.multiply(X_mask,(currentD-lastD))
    D_error = np.linalg.norm(D_error_mat)
    DR_error_mat = np.multiply(X_mask,(currentD-R))
    DR_error = np.linalg.norm(DR_error_mat)
    return D_error,DR_error

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

def parse_args():    
    parser = argparse.ArgumentParser(description="This is a template of machine learning developping source code.")
    parser.add_argument('-m', '--m_topofallfeature', type=int, default=1822,help = "number of drugs")
    parser.add_argument('-n', '--n_topofallfeature', type=int, default=1447,help='number of targets')
    parser.add_argument('-alpha', '--alpha_topofallfeature', type=float, default=0.5)
    parser.add_argument('-mu', '--mu_topofallfeature', type=float, nargs='?', default=10,help='regularization penalty')
    parser.add_argument('-lamda', '--lamda_topofallfeature', type=float, default=10)
    parser.add_argument('-delta', '--delta_topofallfeature', type=float, default=0.5)
    parser.add_argument('-data_folder', '--data_folder_topofallfeature', type=str, default="/home/amber/MULGA/dataset/DrugBank/")
    parser.add_argument('-XSD_path', '--XSD_path_topofallfeature', type=str, default="dti_mat.xlsx")
    parser.add_argument('-XSS_path', '--XSS_path_topofallfeature', type=str, default="drug_affinity_mat.xlsx")
    parser.add_argument('-XDD_path', '--XDD_path_topofallfeature', type=str, default="target_affinity_mat.xlsx")
    parser.add_argument('-topk', '--topk_topofallfeature', type=int, default=1)
    return parser.parse_args()   


if __name__=="__main__":    
    epsilon_D = 0.001
    epsilon_DR = 0.001
    args=parse_args()
    m = args.m_topofallfeature
    n = args.n_topofallfeature
    alpha = args.alpha_topofallfeature
    mu= args.mu_topofallfeature 
    lamda=args.lamda_topofallfeature 
    delta=args.delta_topofallfeature
    
    data_folder = args.data_folder_topofallfeature
    XSD_path = args.XSD_path_topofallfeature
    XSS_path = args.XSS_path_topofallfeature
    XDD_path = args.XDD_path_topofallfeature
    topk = args.topk_topofallfeature 
    XSS,XDD,XSD,XSD_mask_arr,known_samples = read_data(data_folder,XSS_path,XDD_path,XSD_path)
    A_unknown_mask = 1 - XSD   
    drug_dissimmat = get_drug_dissimmat(XSS,topk = topk).astype(int)
    kf = KFold(n_splits =10, shuffle=True)      #10 fold
    train_all=[]
    test_all=[]
    for train_ind,test_ind in kf.split(known_samples):  
        train_all.append(train_ind) 
        test_all.append(test_ind) 
    overall_auroc = 0 
    overall_aupr = 0
    overall_f1 = 0 
    overall_acc = 0
    overall_recall = 0    
    overall_specificity = 0
    overall_precision = 0 
    for fold_int in range(10):
        print('fold_int',fold_int)
        XSD_train_id = train_all[fold_int]
        XSD_train = known_samples[XSD_train_id]
        XSD_train_list = np.zeros_like(XSD_mask_arr)
        XSD_train_list[XSD_train] = 1       
        XSD_train_mask = XSD_train_list.reshape((XSD.shape[0],XSD.shape[1]))
        XSD_train_matrix = XSD_train_mask
        X, X_mask = Combine_X(XSD_train_matrix,XSS,XSD_train_mask,XDD)                
        #initialize
        lastD =  lastY = np.multiply(X_mask,X)
        for i in range(100):         
            R = R_update(lamda,mu,X_mask,X,lastY,lastD)
            currentD = D_update(mu,R,lastY)
            currentY = Y_update(lastY,delta,currentD,R)
            D_error,DR_error = Converge_threshold(lastD,currentD,R,X_mask)
            print('D_error',D_error)
            #print('DR_error',DR_error)
            if (D_error < epsilon_D) and (DR_error < epsilon_DR):
                break
            else:
                lastD = currentD
                lastY = currentY
        XSD_hat = currentD[0:m,m:(m+n)]  
        XSD_hat_arr=XSD_hat.flatten()                 #flattten prediction 
        pos_test_id = test_all[fold_int]
        pos_test = known_samples[pos_test_id]
        pos_test_mask_list = np.zeros_like(XSD_mask_arr)
        pos_test_mask_list[pos_test] = 1 
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
    auroc_ = overall_auroc/10
    aupr_ = overall_aupr/10
    f1_ = overall_f1/10
    acc_ = overall_acc/10
    recall_ = overall_recall/10
    specificity_ = overall_specificity/10
    precision_ = overall_precision/10
    print('mean_auroc:',auroc_)
    print('mean_aupr:',aupr_)