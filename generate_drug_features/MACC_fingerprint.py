# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 22:41:45 2022
@author: Jiani Ma

MACCS fingerprint for drugs
input: SMILES
output: MACCS fingerprint with binary values
"""

import pandas as pd 
import numpy as np 
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
import re
import argparse

"""
MACCS feature: 167 length  
Usage: 

python MACC_fingerprint.py -drug_smile_path - feature_save_path 
"""

def get_config():
    parse = argparse.ArgumentParser(description='common train config')    
    # parameter for the data reading and train-test-splitting
    parse.add_argument('-drug_smile_path', '--drug_smile_path_topofallfeature', type=str, nargs='?', default="/home/jiani.ma/MULGA/dataset/KIBA/drugs.xlsx",help="root dataset path")
    parse.add_argument('-feature_save_path', '--feature_save_path_topofallfeature', type=str, nargs='?', default="/home/jiani.ma/MULGA/dataset/KIBA//MACC_features.xlsx",help="setting the dataset:drugbank, davis or kiba")
    config = parse.parse_args()
    return config


def cal_fingerprint(drug_smile_path,feature_save_path):
    drug_smile_df = pd.read_excel(drug_smile_path)
    smiles = drug_smile_df["SMILES"].values
    smiles = list(smiles)
    drug_macc_feature = np.zeros((len(smiles),167))

    for i in range(0,len(smiles)):
        print("i:",i)
        m = smiles[i]
        mol = Chem.MolFromSmiles(m)
        macc_ = MACCSkeys.GenMACCSKeys(mol).ToBitString()
        finger_str = re.findall(r'\w{1}',macc_)
        finger_str_ = np.array(finger_str,dtype=int)
        drug_macc_feature[i] = finger_str_
    df = pd.DataFrame(drug_macc_feature)
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
# drug_macc_feature = np.zeros((len(smiles),167))

# for i in range(0,len(smiles)):
#     print("i:",i)
#     m = smiles[i]
#     mol = Chem.MolFromSmiles(m)
#     macc_ = MACCSkeys.GenMACCSKeys(mol).ToBitString()
#     finger_str = re.findall(r'\w{1}',macc_)
#     finger_str_ = np.array(finger_str,dtype=int)
#     drug_macc_feature[i] = finger_str_
    
# df = pd.DataFrame(drug_macc_feature)
# df.to_excel("D:/11_DTI_work/DrugBank/data/Finally/MACC_chemical_property.xlsx")

