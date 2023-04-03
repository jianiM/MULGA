"""
@author: jiani ma
"""
import argparse
def get_config():
    parse = argparse.ArgumentParser(description='common train config')
    # parameters for the drug target encoding 
    parse.add_argument('-max_drug_len', '--max_drug_len_topofallfeature', type=int, default=100)
    parse.add_argument('-max_target_len', '--max_target_len_topofallfeature', type=int, default=1000)
    parse.add_argument('-root_path', '--root_path_topofallfeature', type=str, default="/home/jiani.ma/MULGA/dataset",help="root paht for dataset")
    parse.add_argument('-dataset', '--dataset_topofallfeature', type=str, default="KIBA")
    parse.add_argument('-drug_file', '--drug_file_topofallfeature', type=str, default="drugs.xlsx")
    parse.add_argument('-target_file', '--target_file_topofallfeature', type=str, default="targets.xlsx")
    # parameters for main 
    parse.add_argument('-drug_num', '--drug_num_topofallfeature', type=int, default=1720)
    parse.add_argument('-target_num', '--target_num_topofallfeature', type=int, default=220)
    parse.add_argument('-drug_kernel_size', '--drug_kernel_size_topofallfeature', type=int, default=5)
    parse.add_argument('-target_kernel_size', '--target_kernel_size_topofallfeature', type=int, default=11)
    parse.add_argument('-num_filters', '--num_filters_topofallfeature', type=int, default=3)
    parse.add_argument('-embedding_size', '--embedding_size_topofallfeature', type=int, default=128)
    parse.add_argument('-dropout', '--dropout_topofallfeature', type=float, default=0.5)
    parse.add_argument('-batch_size', '--batch_size_topofallfeature', type=int, default=256)
    parse.add_argument('-topk', '--topk_topofallfeature', type=int, default=1)
    parse.add_argument('-lr', '--lr_topofallfeature', type=float, nargs='?', default=0.000001)
    parse.add_argument('-drug_sim_file', '--drug_sim_file_topofallfeature', type=str, default='drug_affinity_mat.xlsx')
    parse.add_argument('-target_sim_file', '--target_sim_file_topofallfeature', type=str, default='target_affinity_mat.xlsx')
    parse.add_argument('-dti_file', '--dti_file_topofallfeature', type=str,default='dti_mat.xlsx')
    parse.add_argument('-drug_encoder_path', '--drug_encoder_path_topofallfeature', type=str, default="/home/jiani.ma/MULGA/DeepDTA/KIBA_encoder/drug_encoder.xlsx")
    parse.add_argument('-target_encoder_path', '--target_encoder_path_topofallfeature', type=str, default="/home/jiani.ma/MULGA/DeepDTA/KIBA_encoder/target_encoder.xlsx")
    parse.add_argument('-device', '--device_topofallfeature', type=str, nargs='?', default="cuda:1",help="setting the cuda device")
    parse.add_argument('-n_splits', '--n_splits_topofallfeature', type=int, nargs='?', default=10,help="k fold")
    parse.add_argument('-epoch_num', '--epoch_num_topofallfeature', type=int, default=25)
    config = parse.parse_args()
    return config