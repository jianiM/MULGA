# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 09:47:10 2022

@author: Jiani Ma

Mogan fingerprint for drugs

"""
import pandas as pd 
import numpy as np 
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
import re
import argparse


def get_config():
    parse = argparse.ArgumentParser(description='common train config')    
    # parameter for the data reading and train-test-splitting
    parse.add_argument('-drug_smile_path', '--drug_smile_path_topofallfeature', type=str, nargs='?', default="/home/jiani.ma/MULGA/dataset/KIBA/drugs.xlsx",help="root dataset path")
    parse.add_argument('-feature_save_path', '--feature_save_path_topofallfeature', type=str, nargs='?', default="/home/jiani.ma/MULGA/dataset/KIBA//Mogan_features.xlsx",help="setting the dataset:drugbank, davis or kiba")
    config = parse.parse_args()
    return config

def cal_fingerprint(drug_smile_path,feature_save_path):
    drug_smile_df = pd.read_excel(drug_smile_path)
    smiles = drug_smile_df["SMILES"].values

    smiles = list(smiles)
    nBits = 512
    drug_mogan_fingerprint = np.zeros((len(smiles),nBits)) 

    for i in range(len(smiles)):
        m = smiles[i]
        m = Chem.MolFromSmiles(m)
        fp = AllChem.GetMorganFingerprintAsBitVect(m,3,nBits=nBits).ToBitString()
        finger_str = re.findall(r'\w{1}',fp)
        finger_str_ = np.array(finger_str,dtype=int)
        drug_mogan_fingerprint[i] = finger_str_
    df = pd.DataFrame(drug_mogan_fingerprint)
    df.to_excel(feature_save_path)


    
if __name__ == "__main__":
    config = get_config()     
    drug_smile_path = config.drug_smile_path_topofallfeature
    feature_save_path = config.feature_save_path_topofallfeature 
    cal_fingerprint(drug_smile_path,feature_save_path) 







# drug_smile_path = "D:/11_DTI_work/DrugBank/data/Finally/drug_chemical_property.xlsx"
# drug_smile_df = pd.read_excel(drug_smile_path)
# smiles = drug_smile_df["SMILES"].values

# smiles = list(smiles)
# nBits = 512
# drug_mogan_fingerprint = np.zeros((len(smiles),nBits)) 

# for i in range(len(smiles)):
#     m = smiles[i]
#     m = Chem.MolFromSmiles(m)
#     fp = AllChem.GetMorganFingerprintAsBitVect(m,3,nBits=nBits).ToBitString()
#     finger_str = re.findall(r'\w{1}',fp)
#     finger_str_ = np.array(finger_str,dtype=int)
#     drug_mogan_fingerprint[i] = finger_str_


# df = pd.DataFrame(drug_mogan_fingerprint)
# df.to_excel("D:/11_DTI_work/DrugBank/data/Finally/mogan_feature.xlsx")