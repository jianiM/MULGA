# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 11:44:01 2023

@author: amber
"""


"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd 
import numpy as np 
import os 
import argparse

def get_config():
    parse = argparse.ArgumentParser(description='common train config')
    # parameters for the data reading and train-test-splitting
    parse.add_argument('-root_path', '--root_path_topofallfeature', type=str, default="/home/jiani.ma/MULGA/generate_protein_features/protein_features/")
    parse.add_argument('-ctdc_path', '--ctdc_path_topofallfeature', type=str, default="CTDC_features.xlsx")
    parse.add_argument('-ctdt_path', '--ctdt_path_topofallfeature', type=str, default="CTDT_features.xlsx")
    parse.add_argument('-ctdd_path', '--ctdd_path_topofallfeature', type=str, default="CTDD_features.xlsx") 
    parse.add_argument('-out', '--out_topofallfeature', type=str, default="CTD_feature.xlsx")
    config = parse.parse_args()
    return config


def concate_ctd(root_path,ctdc_path,ctdt_path,ctdd_path,out):
    CTDC_path = os.path.join(root_path,ctdc_path) 
    CTDC = pd.read_excel(CTDC_path)
    CTDT_path = os.path.join(root_path,ctdt_path)
    CTDT = pd.read_excel(CTDT_path)
    CTDD_path = os.path.join(root_path,ctdd_path)
    CTDD = pd.read_excel(CTDD_path)
    CTD = pd.concat([CTDC,CTDT,CTDD],axis=1)
    out_path = os.path.join(root_path,out)
    CTD.to_excel(out_path)
    
    

if __name__ == "__main__": 
    config = get_config()
    root_path = config.root_path_topofallfeature
    ctdc_path = config.ctdc_path_topofallfeature
    ctdt_path = config.ctdt_path_topofallfeature
    ctdd_path = config.ctdd_path_topofallfeature 
    out = config.out_topofallfeature
    concate_ctd(root_path,ctdc_path,ctdt_path,ctdd_path,out)
    
    
