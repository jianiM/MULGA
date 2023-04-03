# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 17:03:50 2022

@author: Jiani Ma
"""
import pandas as pd 
import sys, re, math, time
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import collections
from collections import OrderedDict
from matplotlib.pyplot import cm
from config_init import get_config
import os 

"""
protein sequence encoding  
"""
CHARPROTSET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
				"U": 19, "T": 20, "W": 21, 
				"V": 22, "Y": 23, "X": 24, 
				"Z": 25 }
CHARPROTLEN = 25

"""
drug sequence encoding 
"""

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2, 
				"1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6, 
				"9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43, 
				"D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13, 
				"O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51, 
				"V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56, 
				"b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60, 
				"l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64


def label_smiles(line, MAX_SMI_LEN, smi_ch_ind):
	X = np.zeros(MAX_SMI_LEN)
	for i, ch in enumerate(line[:MAX_SMI_LEN]): #	x, smi_ch_ind, y
		X[i] = smi_ch_ind[ch]
	return X 

def label_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
	X = np.zeros(MAX_SEQ_LEN)
	for i, ch in enumerate(line[:MAX_SEQ_LEN]):
		X[i] = smi_ch_ind[ch]
	return X 

if __name__=="__main__":
    config = get_config()
    max_drug_len = config.max_drug_len_topofallfeature
    max_target_len = config.max_target_len_topofallfeature
    root_path = config.root_path_topofallfeature
    dataset = config.dataset_topofallfeature
    drug_file = config.drug_file_topofallfeature
    target_file = config.target_file_topofallfeature 
    drug_encoder_file = config.drug_encoder_file_topofallfeature
    target_encoder_file = config.target_encoder_file_topofallfeature 

    drug_smile_path = os.path.join(root_path,dataset,drug_file)
    protein_fasta_path = os.path.join(root_path,dataset,target_file) 
    #drug_encoder_path = os.path.join(root_path,dataset,drug_encoder_file)
    #target_encoder_path = os.path.join(root_path,dataset,target_encoder_file)

    # drug smile encode 
    drug_smile_li = pd.read_excel(drug_smile_path)['SMILES'].values
    drug_smile_li = list(drug_smile_li)  
    drug_smile_encoder = np.zeros((len(drug_smile_li),max_drug_len))  
    for i in range(len(drug_smile_li)):
        drug_label = label_smiles(drug_smile_li[i], max_drug_len, CHARISOSMISET) 
        drug_smile_encoder[i] = drug_label 
    drug_smile_encoder = pd.DataFrame(drug_smile_encoder)
    drug_smile_encoder.to_excel(drug_encoder_file)
    
    #target fasta encode
    target_smile_li = pd.read_excel(protein_fasta_path)['Fasta'].values
    target_smile_li = list(target_smile_li) 
    target_fasta_encoder = np.zeros((len(target_smile_li),max_target_len))    
    for j in range(len(target_smile_li)):
        target_label = label_sequence(target_smile_li[j], max_target_len, CHARPROTSET) 
        target_fasta_encoder[j] = target_label 
    target_fasta_encoder_df = pd.DataFrame(target_fasta_encoder)
    target_fasta_encoder_df.to_excel(target_encoder_file)








