# -*- coding: utf-8 -*-

"""
Created on Sun May 29 17:55:57 2022
@author: Jiani Ma
"""
import numpy as np 
import pandas as pd 
#from data_reading import read_data 
from sklearn.model_selection import KFold
from utils import Construct_G,Construct_H, one_hot_tensor,Normalize_adj
import torch 
import torch.nn as nn 
import torch.nn.functional as F



def kf_split(known_sample,n_splits):
    kf = KFold(n_splits, shuffle=True)      #10 fold
    train_all=[]
    test_all=[]
    for train_ind,test_ind in kf.split(known_sample):  
        train_all.append(train_ind) 
        test_all.append(test_ind)
    return train_all, test_all



