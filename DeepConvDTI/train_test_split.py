# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 15:22:07 2022

@author: Jiani Ma
"""

import numpy as np 
import pandas as pd 
from sklearn.model_selection import KFold



def kf_split(known_sample,n_splits):
    kf = KFold(n_splits, shuffle=True)      #10 fold
    train_all=[]
    test_all=[]
    for train_ind,test_ind in kf.split(known_sample):  
        train_all.append(train_ind) 
        test_all.append(test_ind)
    return train_all, test_all



