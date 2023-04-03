# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 12:26:58 2022
@author: Jiani Ma
"""

import pandas as pd 
import numpy as np  
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

"""
input tensor: drug_feature [batch_size,text_len] 
drug_dict_len=25
embedding_size=128
num_filters=32
drug_kernel_size=4 
# drug_kernel_size = 2 的时候，不改变size
[batch_size,num_filter*3,text_len]
"""

class DrugBlock(nn.Module):
    def __init__(self,drug_dict_len,embedding_size,num_filters,drug_kernel_size):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings = drug_dict_len,embedding_dim = embedding_size)   #(25,128)
        self.conv1_layer = nn.Conv1d(in_channels=embedding_size,out_channels=num_filters,kernel_size =drug_kernel_size, padding=0,stride=1)
        self.conv2_layer = nn.Conv1d(in_channels=num_filters,out_channels=2*num_filters,kernel_size =drug_kernel_size, padding=0,stride=1)
        self.conv3_layer = nn.Conv1d(in_channels=2*num_filters,out_channels=3*num_filters,kernel_size =drug_kernel_size, padding=0,stride=1)    
        self.maxpool_layer = nn.MaxPool1d(kernel_size=2, stride=2)        
    def forward(self,drug_feature):
        drug_enconding = self.embedding_layer(drug_feature)         #(256,100) -- (64,128) embedding ---> (256,100,128)
        drug_encoding = drug_enconding.permute(0,2,1)               # (256,128,100)
        drug_encoding = self.conv1_layer(drug_encoding)             # (256,32,97)
        drug_encoding = F.relu(drug_encoding)
        drug_encoding = self.conv2_layer(drug_encoding)             # (256,64,94)
        drug_encoding = F.relu(drug_encoding)
        drug_encoding = self.conv3_layer(drug_encoding)             # (256,96,91)
        drug_encoding = F.relu(drug_encoding)
        drug_encoding = self.maxpool_layer(drug_encoding)           # (256,96,45)
        return drug_encoding


class TargetBlock(nn.Module): 
    def __init__(self,target_dict_len,embedding_size,num_filters,target_kernel_size):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=target_dict_len,embedding_dim=embedding_size)
        self.conv1_layer = nn.Conv1d(in_channels=embedding_size,out_channels=num_filters,kernel_size=target_kernel_size,padding=0,stride=1)
        self.conv2_layer = nn.Conv1d(in_channels=num_filters,out_channels=2*num_filters,kernel_size=target_kernel_size,padding=0,stride=1)
        self.conv3_layer = nn.Conv1d(in_channels=2*num_filters,out_channels=3*num_filters,kernel_size=target_kernel_size,padding=0,stride=1)
        self.maxpool_layer = nn.MaxPool1d(kernel_size=2, stride=2)    
    def forward(self,target_feature):   
        target_encoding = self.embedding_layer(target_feature)  
        target_encoding = target_encoding.permute(0,2,1)        
        target_encoding = self.conv1_layer(target_encoding)      
        target_encoding = F.relu(target_encoding)
        target_encoding = self.conv2_layer(target_encoding)      #(256,64,986)
        target_encoding = F.relu(target_encoding)
        target_encoding = self.conv3_layer(target_encoding)      #(256,96,979)
        target_encoding = F.relu(target_encoding)
        target_encoding = self.maxpool_layer(target_encoding)    #(256,96,489)
        return target_encoding

class DeepDTA(nn.Module):
    def __init__(self,drug_dict_len,target_dict_len,embedding_size,num_filters,drug_kernel_size,target_kernel_size,fc_dim,dropout):
        super().__init__()
        self.dropout = dropout
        self.drugCNN = DrugBlock(drug_dict_len,embedding_size,num_filters,drug_kernel_size)
        self.targetCNN = TargetBlock(target_dict_len,embedding_size,num_filters,target_kernel_size)
        self.fc1 = nn.Linear(in_features = fc_dim[0],out_features = fc_dim[1]) 
        self.fc2 = nn.Linear(in_features = fc_dim[1],out_features = fc_dim[2])
        self.fc3 = nn.Linear(in_features = fc_dim[2],out_features = fc_dim[3])
        
    def forward(self,drug_feature,target_feature,batch_size):
        drug_feature = self.drugCNN(drug_feature)
        target_feature = self.targetCNN(target_feature)
        drug_target_feature = torch.cat((drug_feature,target_feature),axis=-1)  #[512,96,534]
        drug_target_feature = drug_target_feature.view((batch_size,drug_target_feature.size(-1)*drug_target_feature.size(-2)))               
        y = self.fc1(drug_target_feature)
        y = F.relu(y)
        y = F.dropout(y,self.dropout)
        y = self.fc2(y)
        y = F.relu(y)
        y = F.dropout(y,self.dropout)
        y = self.fc3(y)
        return y 



