# -*- coding: utf-8 -*-
"""
Created on Sun May 29 17:55:57 2022
@author: Jiani Ma
"""

import numpy as np 
import torch
import torch.nn.functional as F 
#torch.manual_seed(args.seed)
from sklearn.metrics import roc_curve, auc,average_precision_score,f1_score,accuracy_score



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
    
def loss_function(pos_score,neg_score,drug_num,target_num):
    lamda = neg_score.size(0)/pos_score.size(0)
    term_one = lamda * torch.sum(torch.log(pos_score)) 
    term_two = torch.sum(torch.log(1.0-neg_score))
    term = term_one + term_two 
    coeff = (-1.0)/(drug_num*target_num)
    result = coeff * term 
    return result     
    
def Construct_G(A_train_mat,SR,SD):
    SR_ = np.zeros_like(SR)
    SD_ = np.zeros_like(SD)
    A_row1 = np.hstack((SR_,A_train_mat))
    A_row2 = np.hstack((A_train_mat.T,SD_))
    G = np.vstack((A_row1,A_row2))    
    G = torch.FloatTensor(G)
    G = Normalize_adj(G)
    return G

def Construct_H(A_train_mat,SR,SD): 
    H_row1 = np.hstack((SR,A_train_mat))
    H_row2 = np.hstack((A_train_mat.T,SD))
    H = np.vstack((H_row1,H_row2))
    H = torch.FloatTensor(H)
    return H 
        
def SnLaplacianMatrix(X,n): 
    W = np.zeros((n,n))
    D = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            W[i][j] = np.dot(X[:,i],X[:,j])
    d = np.sum(W,axis = 1)   #row sum 
    D = np.diag(d) 
    # normlized_d = 1.0/np.sqrt(d)
    # normlized_D = np.diag(normlized_d)
    snL = D - W 
    # snL_tmp = np.dot(normlized_D, L) 
    # snL = np.dot(snL_tmp, normlized_D)
    return snL

def one_hot_tensor(y, num_dim):
    y_onehot = torch.zeros(y.shape[0], num_dim)
    y_onehot.scatter_(1, y.view(-1,1), 1)
    return y_onehot

# transform non-symmetric adjacency to symmetric ones
def Normalize_adj(adj):
    adj_T = adj.transpose(0,1)
    I = torch.eye(adj.shape[0])
    adj = adj + adj_T*(adj_T > adj).float() - adj*(adj_T > adj).float()
    adj = F.normalize(adj + I, p=1)
    return adj

def Global_Normalize(mat):
    max_val = np.max(mat)
    min_val = np.min(mat)
    mat = (mat-min_val)/(max_val-min_val)
    return mat
