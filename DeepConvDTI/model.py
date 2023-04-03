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
        self.fc1 = nn.Linear(in_features=12800, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=32)
           
    def forward(self,batch_size,drug_feature):
        drug_enconding = self.embedding_layer(drug_feature)         #(512,100) -- (64,128) embedding ---> (512,100,128)
        drug_encoding = drug_enconding.permute(0,2,1)               #(512,128,100)
        drug_encoding = drug_encoding.reshape((batch_size,-1))         # (512,12800)
        drug_encoding = self.fc1(drug_encoding)                     #(512,1024)
        drug_encoding = F.relu(drug_encoding)
        drug_encoding = self.fc2(drug_encoding)                     #(512,256)
        drug_encoding = F.relu(drug_encoding)
        drug_encoding = self.fc3(drug_encoding)                     #(512,32)
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
        target_encoding = self.conv2_layer(target_encoding)      #(512,6,980)
        target_encoding = F.relu(target_encoding)
        target_encoding = self.conv3_layer(target_encoding)      #(512,9,970)
        target_encoding = F.relu(target_encoding)
        target_encoding = self.maxpool_layer(target_encoding)    #(512,9,485)
        return target_encoding

class DeepConvDTI(nn.Module):
    def __init__(self,drug_dict_len,target_dict_len,embedding_size,num_filters,drug_kernel_size,target_kernel_size,fc_dim,dropout):
        super().__init__()
        self.dropout = dropout
        self.drugCNN = DrugBlock(drug_dict_len,embedding_size,num_filters,drug_kernel_size)    # 512 *32
        self.targetCNN = TargetBlock(target_dict_len,embedding_size,num_filters,target_kernel_size) #(512,9,485)
        self.fc1 = nn.Linear(in_features = fc_dim[0],out_features = fc_dim[1]) 
        self.fc2 = nn.Linear(in_features = fc_dim[1],out_features = fc_dim[2])
        self.fc3 = nn.Linear(in_features = fc_dim[2],out_features = fc_dim[3])
       
    def forward(self,drug_feature,target_feature,batch_size):
        drug_feature = self.drugCNN(batch_size,drug_feature)                      # 512*32
        target_feature = self.targetCNN(target_feature)                           #(512,9,485)
        target_feature = target_feature.reshape((batch_size,-1))                  # (512, 9*485)           
        drug_target_feature = torch.cat((drug_feature,target_feature),axis=-1)    #[512, 9*485+32]
        y = self.fc1(drug_target_feature)
        y = F.relu(y)
        y = F.dropout(y,self.dropout)
        y = self.fc2(y)
        y = F.relu(y)
        y = F.dropout(y,self.dropout)
        y = self.fc3(y)
        return y 



