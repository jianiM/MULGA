# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 10:03:42 2022
@author: Jiani Ma
"""

import biotite.database.entrez as entrez
import biotite.sequence as seq
import biotite.sequence.io.fasta as fasta
import os
import pandas as pd 
import argparse





def get_config():
    parse = argparse.ArgumentParser(description='common train config')    
    parse.add_argument('-protein_path', '--protein_path_topofallfeature', type=str, default="/home/jiani.ma/MULGA/dataset/KIBA/targets.xlsx")
    parse.add_argument('-path_to_directory', '--path_to_directory_topofallfeature', type=str, default="/home/jiani.ma/MULGA/generate_protein_features/fasta_seqs")
    config = parse.parse_args()
    return config



def extract_fasta(protein_path,path_to_directory):
    proteins = pd.read_excel(protein_path)["Target"].values
    protein_li = list(proteins)
    file_name = entrez.fetch(protein_li,path_to_directory,suffix="fa",db_name="protein",ret_type="fasta")
    for value in file_name:
        fasta_file = fasta.FastaFile()
        fasta_file.read(value)
        print(fasta_file)
    

    
    
    
if __name__=="__main__":

    config = get_config()
    
    protein_path = config.protein_path_topofallfeature
    path_to_directory = config.path_to_directory_topofallfeature
    
    print('protein_path:',protein_path)
    print('path_to_directory:',path_to_directory)
    
    extract_fasta(protein_path,path_to_directory)
    
    
    
        