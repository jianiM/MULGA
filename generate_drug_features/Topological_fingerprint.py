# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 10:34:10 2022

@author: Jiani Ma


From what I can gather the RDKFingerprint is a "Daylight-like" substructure fingerprint that uses a 
bit vector where each bit is set by the presence of a particular substructure within a molecule. 
The default settings (maxPath default=7) consider substructures that are a maximum of 7 bonds long. 
As there is no predefined substructure set, it is impossible to set a bit for each existing pattern 
so each key is considered as a seed to a pseudo-random number generator ('hashing'). 
The output of this is a set of bits (nBitsPerHash, default=2) with numbers between 0 and fpSize default=2048 
which is used to set the corresponding bits in the fingerprint.
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
    parse.add_argument('-feature_save_path', '--feature_save_path_topofallfeature', type=str, nargs='?', default="/home/jiani.ma/MULGA/dataset/KIBA/drug_features/Topological_features.xlsx",help="setting the dataset:drugbank, davis or kiba")
    config = parse.parse_args()
    return config

def cal_fingerprint(drug_smile_path,feature_save_path):
    drug_smile_df = pd.read_excel(drug_smile_path)
    smiles = drug_smile_df["SMILES"].values
    smiles = list(smiles)
    fpsize = 512
    drug_rdkfingerprint = np.zeros((len(smiles),fpsize))
    for i in range(len(smiles)):
        print("converting:",i)
        m = smiles[i] 
        mol = Chem.MolFromSmiles(m)        # rdkit has it's inner molecular data representation object: mol    
        fp = Chem.RDKFingerprint(mol,fpSize=fpsize).ToBitString()
        finger_str = re.findall(r'\w{1}',fp)
        finger_str_ = np.array(finger_str,dtype=int)
        drug_rdkfingerprint[i] = finger_str_
                
    df = pd.DataFrame(drug_rdkfingerprint)
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
# fpsize = 512

# drug_rdkfingerprint = np.zeros((len(smiles),fpsize))

# for i in range(len(smiles)):
#     print("i",i)
#     m = smiles[i] 
#     mol = Chem.MolFromSmiles(m)        # rdkit has it's inner molecular data representation object: mol    
#     fp = Chem.RDKFingerprint(mol,fpSize=fpsize).ToBitString()
#     finger_str = re.findall(r'\w{1}',fp)
#     finger_str_ = np.array(finger_str,dtype=int)
#     drug_rdkfingerprint[i] = finger_str_
            
# df = pd.DataFrame(drug_rdkfingerprint)
# df.to_excel("D:/11_DTI_work/DrugBank/data/Finally/drug_RDK_feature.xlsx")
    