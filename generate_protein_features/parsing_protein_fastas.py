# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 11:07:40 2022

@author: Jiani Ma
"""
import os 
import pandas as pd
import re
import timeit
import argparse





def get_config():
    parse = argparse.ArgumentParser(description='common train config')    
    parse.add_argument('-root_path', '--root_path_topofallfeature', type=str, default="/home/jiani.ma/MULGA/generate_protein_features/fasta_file")
    parse.add_argument('-protein_list_path', '--protein_list_path_topofallfeature', type=str, default="/home/jiani.ma/MULGA/dataset/KIBA/targets.xlsx")
    config = parse.parse_args()
    return config



def parsing_fasta(root_path,protein_list_path):
    pattern_title = re.compile(r'^>.*', re.M)
    pattern_n = re.compile(r'\n')
    protein_pd = pd.read_excel(protein_list_path)['Target']
    protein_list = list(protein_pd.values)
    for i in range(len(protein_list)):
        protein_name = protein_list[i]+'.fa'
        file_path = os.path.join(root_path,protein_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        data = re.split(pattern_title, text)
        data2 = [re.sub(pattern_n, '', i) for i in data]
        fasta_sequence = data2[1]
        print('>'+protein_list[i])
        print(fasta_sequence)

if __name__ == "__main__": 

    config = get_config()
    root_path = config.root_path_topofallfeature
    protein_list_path = config.protein_list_path_topofallfeature
    parsing_fasta(root_path,protein_list_path)




