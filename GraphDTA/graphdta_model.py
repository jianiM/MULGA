# -*- coding: utf-8 -*-
"""
Created on Sun May 29 17:55:57 2022
@author: Jiani Ma
"""
import torch 
import torch.nn as nn 
import torch.nn.functional as F


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)

class GraphConvolution(nn.Module):
    def __init__(self,in_feature,out_feature,bias=True):
        super().__init__()
        self.in_feature = in_feature 
        self.out_feature = out_feature 
        self.weight = nn.Parameter(torch.FloatTensor(in_feature, out_feature))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_feature))
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)
            
    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output + self

class DrugBlock(nn.Module):
    def __init__(self,hgcn_dim,dropout):
        super().__init__()
        self.dropout = dropout 
        self.gc1 = GraphConvolution(hgcn_dim,hgcn_dim*2)   # w1 = (m+n) * k
        self.gc2 = GraphConvolution(hgcn_dim*2,hgcn_dim*4) # w2 = k * k  
        self.gc_fc1 = nn.Linear(hgcn_dim*4,512)  
        self.gc_fc2 = nn.Linear(512,128) 

    def forward(self,feature_mat,adjacent_mat):
        H = self.gc1(feature_mat,adjacent_mat) 
        H = F.leaky_relu(H,0.25)
        H = F.dropout(H, self.dropout, training=True)
        H = self.gc2(H,adjacent_mat)
        H = F.leaky_relu(H,0.25)
        H = self.gc_fc1(H)
        H = F.leaky_relu(H,0.25)
        H = self.gc_fc2(H)
        H = F.leaky_relu(H,0.25) 
        pooling = torch.max(H,dim=0).values     
        pooling = torch.unsqueeze(pooling,dim=0)
        return pooling

class TargetBlock(nn.Module): 
    def __init__(self,target_dict_len,embedding_dim,num_filters,target_kernel_size):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=target_dict_len+1,embedding_dim=embedding_dim)
        self.conv1_layer = nn.Conv1d(in_channels=1000,out_channels=num_filters,kernel_size=target_kernel_size,padding=0,stride=1)
        self.fc1_xt = nn.Linear(num_filters*126, 128)
                
    def forward(self,prot_feature,num_filters):           
        embedded_xt = self.embedding_layer(prot_feature)
        conv_xt = self.conv1_layer(embedded_xt)
        xt = conv_xt.view(-1, num_filters * 126)
        xt = self.fc1_xt(xt)        
        return xt        
        
class GraphDTA(nn.Module):
    def __init__(self,hgcn_dim,dropout,target_dict_len,embedding_dim,num_filters,target_kernel_size):
        super().__init__()
        self.drugGCN = DrugBlock(hgcn_dim,dropout)
        self.targetCNN = TargetBlock(target_dict_len,embedding_dim,num_filters,target_kernel_size)
        self.fc1 = nn.Linear(2*128,128)
        self.fc2 = nn.Linear(128,1)
        
    def forward(self,feature_mat,adjacent_mat,prot_feature,num_filters):   
        drug_feature = self.drugGCN(feature_mat,adjacent_mat)
        target_feature = self.targetCNN(prot_feature,num_filters)
        xc = torch.cat((drug_feature, target_feature), 1)
        xc = self.fc1(xc)
        xc = F.leaky_relu(xc,0.25)
        xc = self.fc2(xc)
        result = torch.sigmoid(xc)
        return result




