"""
@author: jiani ma
"""
import argparse
def get_config():
    parse = argparse.ArgumentParser(description='common train config')
    # parameters for the data reading and train-test-splitting
    parse.add_argument('-root_path', '--root_path_topofallfeature', type=str, nargs='?', default="/home/jiani.ma/MULGA/dataset/",help="root dataset path")
    parse.add_argument('-dataset', '--dataset_topofallfeature', type=str, nargs='?', default="DrugBank",help="setting the dataset:DrugBank or KIBA")

    # parameters for generating drug affinity matrix 
    parse.add_argument('-drug_macc_file', '--drug_macc_file_topofallfeature', type=str, default="drug_features/MACC_features.xlsx")
    parse.add_argument('-drug_mogan_file', '--drug_mogan_file_topofallfeature', type=str, default="drug_features/Mogan_features.xlsx")
    parse.add_argument('-drug_rdk_file', '--drug_rdk_file_topofallfeature', type=str, default="drug_features/Topological_features.xlsx")

    # parameters for generating target affinity matrix 
    parse.add_argument('-AAC_file', '--AAC_file_topofallfeature', type=str, default="protein_features/AAC_features.xlsx")
    parse.add_argument('-CTD_file', '--CTD_file_topofallfeature', type=str, default="protein_features/CTD_feature.xlsx")
    parse.add_argument('-Moran_file', '--Moran_file_topofallfeature', type=str, default="protein_features/Moran_features.xlsx")
    parse.add_argument('-PAAC_file', '--PAAC_file_topofallfeature', type=str, default="protein_features/PAAC_features.xlsx")


    # parameters for the model  
    parse.add_argument('-device', '--device_topofallfeature', type=str, nargs='?', default="cuda:0",help="setting the cuda device")
    parse.add_argument('-n_splits', '--n_splits_topofallfeature', type=int, nargs='?', default= 10,help="k fold")
    
    parse.add_argument('-drug_sim_file', '--drug_sim_file_topofallfeature', type=str, nargs='?', default="drug_affinity_mat.xlsx",help="setting the drug similarity file")
    parse.add_argument('-target_sim_file', '--target_sim_file_topofallfeature', type=str, nargs='?', default="target_affinity_mat.xlsx",help="setting the target similarity file")
    parse.add_argument('-dti_mat', '--dti_mat_topofallfeature', type=str, nargs='?', default="dti_mat.xlsx",help="setting the dti matrix file")
    
    parse.add_argument('-hgcn_dim', '--hgcn_dim_topofallfeature', type=int, nargs='?', default=2500,help='defining the size of hidden layer of GCN.')
    parse.add_argument('-dropout', '--dropout_topofallfeature', type=float, nargs='?', default=0.3,help='ratio of drop the graph nodes.')
    parse.add_argument('-epoch_num', '--epoch_num_topofallfeature', type=int, nargs='?', default=1000,help='number of epoch.')
    parse.add_argument('-lr', '--lr_topofallfeature', type=float, nargs='?', default=0.000005,help='learning rate.')
    parse.add_argument('-topk', '--topk_topofallfeature', type=int, nargs='?', default=1 ,help='ratio of positive samples and negative samples, i.e. 1 or 10.')
    parse.add_argument('-epoch_interv', '--epoch_interv_topofallfeature', type=int, nargs='?', default=10,help='interval for showing the loss')
    config = parse.parse_args()
    return config
