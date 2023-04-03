# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 08:51:08 2022

@author: Jiani Ma
"""
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Variable
from feature_extraction import *
from graphdta_model import GraphConvolution, DrugBlock, TargetBlock,GraphDTA
from utils import *
from torch.utils.data import WeightedRandomSampler,DataLoader
torch.cuda.manual_seed(1223)            
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="This is a template of machine learning developping source code.")
    parser.add_argument('-num_filters', '--num_filters_topofallfeature', type=int, default=6,help='defining the size of hidden layer of GCN.')
    parser.add_argument('-epoch_num', '--epoch_num_topofallfeature', type=int, default=50,help='ratio of drop the graph nodes.')
    parser.add_argument('-lr', '--lr_topofallfeature', type=float, default=0.0001,help='number of epoch.')
    parser.add_argument('-batch_size', '--batch_size_topofallfeature', type=int, default=512)
    parser.add_argument('-device', '--device_topofallfeature', type=str, default='cuda:0')
    parser.add_argument('-dropout', '--dropout_topofallfeature', type=float, default=0.5)
    parser.add_argument('-max_seq_len', '--max_seq_len_topofallfeature', type=int,  default=1000,help='number of epoch.')
    parser.add_argument('-embedding_dim', '--embedding_dim_topofallfeature', type=float, default=128)
    parser.add_argument('-target_kernel_size', '--target_kernel_size_topofallfeature', type=int, default=3)
    parser.add_argument('-topk', '--topk_topofallfeature', type=int, default=10)
    parser.add_argument('-n_splits', '--n_splits_topofallfeature', type=int, default=10)
    parser.add_argument('-data_folder', '--data_folder_topofallfeature', type=str, default="/home/jiani.ma/MULGA/dataset/KIBA")
    parser.add_argument('-drug_sim_path', '--drug_sim_path_topofallfeature', type=str, default="drug_affinity_mat.xlsx",help='number of epoch.')
    parser.add_argument('-target_sim_path', '--target_sim_path_topofallfeature', type=str,default="target_affinity_mat.xlsx",help='number of epoch.')
    parser.add_argument('-DTI_path', '--DTI_path_topofallfeature', type=str, default="dti_mat.xlsx")
    parser.add_argument('-drug_smiles_path', '--drug_smiles_path_topofallfeature', type=str, default='drugs.xlsx')
    parser.add_argument('-target_fasta_path', '--target_fasta_path_topofallfeature', type=str, default='targets.xlsx')
    return parser.parse_args()

if __name__=="__main__":
    hgcn_dim = 78
    target_dict_len= 25
    args = parse_args()
    num_filters = args.num_filters_topofallfeature
    epoch_num = args.epoch_num_topofallfeature
    lr = args.lr_topofallfeature
    device = args.device_topofallfeature
    batch_size = args.batch_size_topofallfeature
    dropout = args.dropout_topofallfeature
    max_seq_len = args.max_seq_len_topofallfeature
    embedding_dim = args.embedding_dim_topofallfeature
    target_kernel_size = args.target_kernel_size_topofallfeature
    topk = args.topk_topofallfeature
    n_splits = args.   n_splits_topofallfeature
    data_folder = args.data_folder_topofallfeature
    drug_sim_path = args.drug_sim_path_topofallfeature
    target_sim_path = args.target_sim_path_topofallfeature
    DTI_path = args.DTI_path_topofallfeature
    drug_smiles_path = args.drug_smiles_path_topofallfeature
    target_fasta_path = args.target_fasta_path_topofallfeature
    SR,A_orig,A_orig_arr,known_sample,drug_smiles,target_fasta = read_data(data_folder,drug_sim_path,DTI_path,drug_smiles_path,target_fasta_path) 
    drug_num = A_orig.shape[0]
    target_num = A_orig.shape[1]
    A_orig_list = A_orig.flatten()         
    drug_dissimmat = get_drug_dissimmat(SR,topk = topk).astype(int)
    seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}    
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
    for fold_int in range(10):
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
        pos_drug_index = pos_train_dti[0]
        pos_target_index = pos_train_dti[1]                                                                  
        pos_labels = torch.ones(len(pos_drug_index),dtype=torch.int64)
    
        A_unknown_mask = 1 - A_orig            
        A_train_mat = A_train_mask    
        train_neg_mask_candidate = get_negative_samples(A_train_mask,drug_dissimmat)
        train_neg_mask = np.multiply(train_neg_mask_candidate, A_unknown_mask)        
        neg_train_dti = np.where(train_neg_mask==1)
        neg_drug_index = neg_train_dti[0].astype(np.int64)
        neg_target_index = neg_train_dti[1].astype(np.int64) 
        neg_labels = torch.zeros(len(neg_drug_index))              
                
        drug_index = np.hstack((pos_drug_index,neg_drug_index))
        target_index = np.hstack((pos_target_index,neg_target_index))
        drug_index = torch.from_numpy(drug_index).long()
        target_index = torch.from_numpy(target_index).long()
        labels = torch.cat((pos_labels,neg_labels))

        train_idx = torch.arange(target_index.size(0))
        model = GraphDTA(hgcn_dim,dropout,target_dict_len,embedding_dim,num_filters,target_kernel_size).to(device)
        
        optimizer = torch.optim.Adam(list(model.parameters()),lr=lr)        
        model.train()
        dataloader = DataLoader(dataset=train_idx,batch_size=batch_size, shuffle=True,drop_last=True)                   
        for epoch in range(epoch_num):       
            print("epoch:",epoch)
            for i,mini_batch_idx in enumerate(dataloader):
                scores = torch.zeros(batch_size,requires_grad=True).to(device)
                batch_drug_indice = drug_index[mini_batch_idx].numpy()
                batch_target_indice = target_index[mini_batch_idx].numpy()        
                batch_labels = labels[mini_batch_idx].numpy()
                batch_pos_label_indice = np.where(batch_labels==1)[0]
                batch_neg_label_indice = np.where(batch_labels==0)[0]
                batch_pos_label_indice = torch.from_numpy(batch_pos_label_indice)
                batch_neg_label_indice = torch.from_numpy(batch_neg_label_indice)                            
                batch_drug_smiles = drug_smiles[batch_drug_indice]   # 256
                batch_target_fastas = target_fasta[batch_target_indice] #256
                for p in range(batch_size):
                    smile =  batch_drug_smiles[p]
                    prot = batch_target_fastas[p]
                    feature_mat, adjacent_mat = drug_graph_construct(smile) 
                    feature_mat = feature_mat.to(device)
                    adjacent_mat = adjacent_mat.to(device)
                    prot_feature = protein_feature_extraction(prot,max_seq_len,seq_dict)          
                    prot_feature = prot_feature.to(device)    
                    y = model(feature_mat,adjacent_mat,prot_feature,num_filters)
                    scores[p] = y
                pos_score = scores[batch_pos_label_indice]
                neg_score = scores[batch_neg_label_indice]          
                loss = loss_function(pos_score,neg_score,batch_size)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                los_  = loss.detach().cpu().item()
            print('los_:',los_)
            
        model.eval()
        pos_test_dti = np.where(A_test_mask==1)
        pos_test_drug_index = pos_test_dti[0]     
        pos_test_target_index = pos_test_dti[1]       
        pos_test_labels = np.ones(len(A_test))
   
        """
        test negative samples
        """
        test_neg_mask_candidate = get_negative_samples(A_test_mask,drug_dissimmat)
        test_neg_mask = np.multiply(test_neg_mask_candidate, A_unknown_mask)           
        neg_test_dti = np.where(test_neg_mask==1)
        neg_test_drug_index = neg_test_dti[0].astype(np.int64)
        neg_test_target_index = neg_test_dti[1].astype(np.int64)        
        neg_test_labels = np.zeros(len(neg_test_drug_index))

        test_drug_index = np.hstack((pos_test_drug_index,neg_test_drug_index))
        test_target_index = np.hstack((pos_test_target_index,neg_test_target_index)) 
        test_smiles = drug_smiles[test_drug_index]
        test_fastas = target_fasta[test_target_index]
        test_scores = np.zeros(len(test_drug_index))
        test_labels = np.hstack((pos_test_labels,neg_test_labels))

        for t in range(len(test_drug_index)):
            test_smile = test_smiles[t]
            test_prot = test_fastas[t]
            test_feature_mat, test_adjacent_mat = drug_graph_construct(test_smile) 
            test_feature_mat = test_feature_mat.to(device)
            test_adjacent_mat= test_adjacent_mat.to(device)
            test_prot_feature = protein_feature_extraction(test_prot,max_seq_len,seq_dict)              
            test_prot_feature = test_prot_feature.to(device)  
            s = model(test_feature_mat,test_adjacent_mat,test_prot_feature,num_filters)  
            test_score = s.detach().cpu().item()
            test_scores[t] = test_score
        TP,FP,FN,TN,fpr,tpr,auroc,aupr,f1_score, accuracy, recall, specificity, precision = get_metric(test_labels,test_scores)
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

    
    
    
