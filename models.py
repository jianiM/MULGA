# -*- coding: utf-8 -*-
"""
Created on Sun May 29 17:55:57 2022
@author: Jiani Ma
"""
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Variable

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

class Decoder(nn.Module):
    def __init__(self,train_W):
        super().__init__()
        self.train_W = train_W  
        
    def forward(self,H,drug_num,target_num):
        HR = H[0:drug_num]
        HD = H[drug_num:(drug_num+target_num)]
        supp1 = torch.mm(HR,self.train_W)
        decoder = torch.mm(supp1,HD.transpose(0,1))    
        return decoder 
        

class GCN_decoder(nn.Module):
    def __init__(self,in_dim,hgcn_dim,train_W,dropout):
        super().__init__()
        self.gc1 = GraphConvolution(in_dim,hgcn_dim)   # w1 = (m+n) * k
        self.gc2 = GraphConvolution(hgcn_dim,hgcn_dim) # w2 = k * k
        self.gc3 = GraphConvolution(hgcn_dim,hgcn_dim) # w3 = k * k       
        self.decoder = Decoder(train_W)     
        self.dropout = dropout 
    
    def forward(self,H,G,drug_num,target_num):
        H = self.gc1(H,G)    
        H = F.leaky_relu(H,0.25)
        H = F.dropout(H, self.dropout, training=True)
        H = self.gc2(H,G)
        H = F.leaky_relu(H,0.25)
        H = F.dropout(H, self.dropout, training=True)
        H = self.gc3(H,G)
        H = F.leaky_relu(H,0.25)           
        decoder = self.decoder(H,drug_num,target_num)        
        
        return decoder





     
    
        
    
    