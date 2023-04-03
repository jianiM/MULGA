# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 15:06:02 2022
@author: Jiani Ma
"""
import os
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import pandas as pd 
import numpy as np 
from model import DrugBlock,TargetBlock,DeepDTA
from utils import get_metric,get_negative_samples,loss_function,one_hot_tensor
from data_reading import read_data,get_drug_dissimmat
from train_test_split import kf_split
from torch.utils.data import WeightedRandomSampler,DataLoader
from sklearn.metrics import roc_curve, auc
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse
torch.cuda.manual_seed(1223)
from config_init import get_config

if __name__=="__main__":
    """
    hyper-parameters
    """
    drug_dict_len = 64 
    target_dict_len = 25
    config = get_config()
    root_path = config.root_path_topofallfeature
    dataset = config.dataset_topofallfeature 
    drug_file = config.drug_file_topofallfeature 
    target_file = config.target_file_topofallfeature
    drug_num = config.drug_num_topofallfeature 
    target_num = config.target_num_topofallfeature 
    drug_kernel_size = config.drug_kernel_size_topofallfeature
    target_kernel_size = config.target_kernel_size_topofallfeature
    num_filters = config.num_filters_topofallfeature
    embedding_size = config.embedding_size_topofallfeature
    dropout = config.dropout_topofallfeature
    batch_size = config.batch_size_topofallfeature 
    topk = config.topk_topofallfeature
    lr = config.lr_topofallfeature
    epoch_num = config.epoch_num_topofallfeature
    
    drug_sim_file = config.drug_sim_file_topofallfeature
    target_sim_file = config.target_sim_file_topofallfeature
    dti_file = config.dti_file_topofallfeature
    drug_encoder_path = config.drug_encoder_path_topofallfeature
    target_encoder_path = config.target_encoder_path_topofallfeature
    device = config.device_topofallfeature
    n_splits = config.n_splits_topofallfeature
    fc_dim = [3*num_filters*529,1024,128,1]
    # data reading 
    SR,A_orig,A_orig_arr,known_sample,drug_encoder_list,target_encoder_list,drug_num,target_num = read_data(root_path,dataset,drug_sim_file,target_sim_file,dti_file,drug_encoder_path,target_encoder_path)
    
    drug_dissimmat = get_drug_dissimmat(SR,topk = topk).astype(int)
    train_all, test_all = kf_split(known_sample,n_splits)    
    negtive_index_arr = np.where(A_orig_arr==0)[0]
    negative_index = torch.LongTensor(negtive_index_arr)    
    overall_auroc = 0 
    overall_aupr = 0
    overall_f1 = 0 
    overall_acc = 0
    overall_recall = 0    
    overall_specificity = 0
    overall_precision = 0 
     
    for fold_int in range(n_splits):
        print('fold_int:',fold_int)        
        A_train_id = train_all[fold_int]
        A_test_id = test_all[fold_int]            
        A_train = known_sample[A_train_id]
        A_test = known_sample[A_test_id]        
        A_train_tensor = torch.LongTensor(A_train)
        A_test_tensor = torch.LongTensor(A_test)        
        A_train_list = np.zeros_like(A_orig_arr)
        A_train_list[A_train] = 1        
        A_test_list = np.zeros_like(A_orig_arr)
        A_test_list[A_test] = 1                                
        A_train_mask = A_train_list.reshape((A_orig.shape[0],A_orig.shape[1]))
        A_test_mask = A_test_list.reshape((A_orig.shape[0],A_orig.shape[1]))

        pos_train_dti = np.where(A_train_mask==1)
        pos_drug_encoder_index = pos_train_dti[0]
        pos_target_encoder_index = pos_train_dti[1]  
        pos_drug_encoder = drug_encoder_list[pos_drug_encoder_index].astype(np.int64)
        pos_target_encoder = target_encoder_list[pos_target_encoder_index].astype(np.int64)                                                          
        pos_labels = torch.ones(len(A_train),dtype=torch.int64)
    
        A_unknown_mask = 1 - A_orig            
        A_train_mat = A_train_mask    
        train_neg_mask_candidate = get_negative_samples(A_train_mask,drug_dissimmat)
        train_neg_mask = np.multiply(train_neg_mask_candidate, A_unknown_mask)
        
        neg_train_dti = np.where(train_neg_mask==1)
        neg_drug_encoder_index = neg_train_dti[0].astype(np.int64)
        neg_target_encoder_index = neg_train_dti[1].astype(np.int64)
        neg_drug_encoder = drug_encoder_list[neg_drug_encoder_index]        
        neg_target_encoder = target_encoder_list[neg_target_encoder_index]          
        neg_labels = torch.zeros(len(neg_drug_encoder_index))              
        
        train_drug_encoder = np.vstack((pos_drug_encoder,neg_drug_encoder))
        train_drug_encoder = train_drug_encoder.astype(np.int64)
        train_target_encoder = np.vstack((pos_target_encoder,neg_target_encoder))
        train_target_encoder = train_target_encoder.astype(np.int64)

        train_drug_encoder = torch.from_numpy(train_drug_encoder)   
        train_drug_encoder = torch.LongTensor(train_drug_encoder)                
        train_target_encoder = torch.from_numpy(train_target_encoder)
        train_target_encoder = torch.LongTensor(train_target_encoder)        
        
        train_encoder_tensor = torch.cat((train_drug_encoder,train_target_encoder),axis=1)#.to(device)
        
        train_idx = torch.arange(train_encoder_tensor.size(0))#.to(device)
        train_labels = torch.cat((pos_labels,neg_labels),axis=0)#.to(device)
        
        pos_test_dti = np.where(A_test_mask==1)
        pos_test_drug_index = pos_test_dti[0]     
        pos_test_target_index = pos_test_dti[1]
        pos_test_drug_encoder = drug_encoder_list[pos_test_drug_index]
        pos_test_target_encoder = target_encoder_list[pos_test_target_index]  
        pos_test_labels = np.ones(len(A_test))

        test_neg_mask_candidate = get_negative_samples(A_test_mask,drug_dissimmat)
        test_neg_mask = np.multiply(test_neg_mask_candidate, A_unknown_mask)           
        neg_test_dti = np.where(test_neg_mask==1)
        neg_test_drug_encoder_index = neg_test_dti[0].astype(np.int64)
        neg_test_target_encoder_index = neg_test_dti[1].astype(np.int64)
        neg_test_drug_encoder = drug_encoder_list[neg_test_drug_encoder_index]        
        neg_test_target_encoder = target_encoder_list[neg_test_target_encoder_index]                         
        neg_test_labels = np.zeros(len(neg_test_drug_encoder_index))
        
        test_drug_encoder = np.vstack((pos_test_drug_encoder,neg_test_drug_encoder))
        test_drug_encoder = test_drug_encoder.astype(np.int64)        
        test_target_encoder = np.vstack((pos_test_target_encoder,neg_test_target_encoder))
        test_target_encoder = test_target_encoder.astype(np.int64)
                
        test_drug_encoder = torch.from_numpy(test_drug_encoder)   
        test_drug_encoder = torch.LongTensor(test_drug_encoder)#.to(device)     
                   
        test_target_encoder = torch.from_numpy(test_target_encoder)
        test_target_encoder = torch.LongTensor(test_target_encoder)#.to(device)                
        test_labels = np.hstack((pos_test_labels,neg_test_labels))
        
        deepdta = DeepDTA(drug_dict_len,target_dict_len,embedding_size,num_filters,drug_kernel_size,target_kernel_size,fc_dim,dropout)#.to(device)
        optimizer = torch.optim.Adam(list(deepdta.parameters()),lr=lr)        
        
        train_dataloader = DataLoader(dataset=train_idx,batch_size=batch_size, shuffle=True,drop_last=True)
        
        deepdta.train()
        for epoch in range(epoch_num):       
            print("epoch:",epoch)
            for i,mini_batch_idx in enumerate(train_dataloader):                
                mini_drug_target_feature = train_encoder_tensor[mini_batch_idx]
                mini_labels = train_labels[mini_batch_idx]               
                mini_labels = torch.tensor(mini_labels,dtype=torch.int64)
                mini_labels_arr = mini_labels  #.cpu().numpy()                
                pos_target_index = np.where(mini_labels_arr==1)[0]
                neg_target_index = np.where(mini_labels_arr==0)[0]                                     
                drug_feature = mini_drug_target_feature[:,0:100]
                target_feature = mini_drug_target_feature[:,100:]
                y_hat = deepdta(drug_feature,target_feature,batch_size)
                logits = torch.sigmoid(y_hat)                
                pos_score = logits[pos_target_index].view(-1)
                neg_score = logits[neg_target_index].view(-1)                
                loss = loss_function(pos_score,neg_score,drug_num,target_num)
                los_ = loss.detach().item()                                
                if los_ < 2e-6:
                    break                 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('loss:',los_)
                
        deepdta.eval()    
        num_test = test_drug_encoder.size(0)
        test_scores = torch.zeros(num_test)      
        for s in range(num_test): 
            test_drug_feature = test_drug_encoder[s].unsqueeze(0) 
            test_target_feature = test_target_encoder[s].unsqueeze(0)          
            predict_score = deepdta(test_drug_feature,test_target_feature,1) 
            test_scores[s] = predict_score        
        test_scores_arr = test_scores.detach().numpy()   #.cpu().numpy()
        test_scores_arr = test_scores_arr.reshape((1,len(test_scores_arr)))[0]  
        TP,FP,FN,TN,fpr,tpr,auroc,aupr,f1_score, accuracy, recall, specificity, precision = get_metric(test_labels,test_scores_arr)
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
    
        

        
        
