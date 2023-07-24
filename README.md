# MULGA
A unified multi-view graph autoencoder-based approach for identifying drug-protein interaction and drug repositioning

![fig1_](https://user-images.githubusercontent.com/87815194/228164057-ead4748a-64c6-482a-8522-ec1e466b7082.png)

# Resources:
+ README.md: this file.

# Datasets 
The datasets we used in this study are provided at https://zenodo.org/deposit/7793559.
### DrugBank
4 files:  
1822 drugs 
1447 proteins
6871 validated DTIs  
* four files: drugs.xlsx     --- drug id and smiles 
              targets.xlsx   --- uniprot id and fasta
              dti_list.xlsx  --- dti mapping list 
              dti_mat.xlsx   --- dti mat 

### KIBA
1720  drugs 
220   proteins
22154 validated DTIs
* four files: drugs.xlsx     --- drug id and smiles 
              targets.xlsx   --- uniprot id and fasta
              dti_list.xlsx  --- dti mapping list 
              dti_mat.xlsx   --- dti mat

### KIBA
1720  drugs 
220   proteins
22154 validated DTIs
* four files: drugs.xlsx     --- drug id and smiles 
              targets.xlsx   --- uniprot id and fasta
              dti_list.xlsx  --- dti mapping list 
              dti_mat.xlsx   --- dti mat 

  

## Step-by-step running for MULGA 
### Prepare conda enviroment and install Python libraries needed
+ conda create -n bio python=3.9 
+ source activate bio 
+ conda install -y -c conda-forge rdkit
+ pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
+ git clone https://github.com/Superzchen/iFeature
+ pip install biotite 

dependencies: 
   + python == 3.9.16 
   + torch == 1.13.1+cu116
   + torchvision == 0.14.1+cu116
   + numpy == 1.24.2 
   + pandas == 1.5.3
   + scikit-learn == 1.2.2 
   + scipy == 1.10.1
   + matplotlib == 3.7.1 
   + seaborn == 0.12.2
   + biotite == 0.33.0

### Usage 
1. generate drug fingerprints for drugs in generate_drug_features packages(take KIBA as example)
   + Running:
   + python MACC_fingerprint.py -drug_smile_path .KIBA/drugs.xlsx -feature_save_path ./KIBA/drug_features/MACC_features.xlsx
   + python Mogan_fingerprint.py -drug_smile_path .KIBA/drugs.xlsx -feature_save_path ./KIBA/drug_features/Mogan_features.xlsx
   + python Topological_fingerprint.py -drug_smile_path .KIBA/drugs.xlsx -feature_save_path ./KIBA/drug_features/Topological_features.xlsx

   ---> drug_features/MACC_features.xlsx
        drug_features/Mogan_features.xlsx
        drug_features/Topological_features.xlsx

2. generate protein features in generate_protein_features packages with iFeatures tool
   + retrieve all the fasta sequences according to the protein uniprot id, and put all the fasta seqs in fasta_file directory. 
   + Running: 
   + python generate_fasta_file.py --protein_path /home/jiani.ma/MULGA/dataset/KIBA/targets.xlsx --path_to_directory /home/jiani.ma/MULGA/generate_protein_features/fasta_seqs
   
   + combing all protein sequences orderly 
   + Running: 
   + python parsing_protein_fastas.py --root_path /home/jiani.ma/MULGA/generate_protein_features/fasta_file --protein_list_path /home/jiani.ma/MULGA/dataset/KIBA/targets.xlsx > fasta_seqs.txt

   ---> fasta_seqs.txt 

   + retrieve protein features including AAC, CTD, Moran Correlation and PAAC features with iFeatures 
   + Running: 
   + python iFeature.py --file ../fasta_seqs.txt --type AAC --out ../protein_features/AAC_features.txt
   + python iFeature.py --file ../fasta_seqs.txt --type CTDC --out ../protein_features/CTDC_features.txt
   + python iFeature.py --file ../fasta_seqs.txt --type CTDT --out ../protein_features/CTDT_features.txt
   + python iFeature.py --file ../fasta_seqs.txt --type CTDD --out ../protein_features/CTDD_features.txt 
   + python CTD_concate.py --ctdc_path --ctdt_path --ctdd_path --out ../CTD_features.txt
   + python iFeature.py --file ../fasta_seqs.txt --type PAAC --out ../protein_features/PAAC_features.txt

   ---> protein_features/AAC_features.xlsx
   ---> protein_features/CTD_features.xlsx
   ---> protein_features/Moran_Correlation_features.xlsx
   ---> protein_features/PAAC_features.xlsx

3. setting all the features path in config_init.py and generate drug affinity matrix and target affinity matrix 
   Running: 
   *python generate_drug_affinity_mat.py > drug_affinity_mat.txt
   *python generate_target_affinity_mat.py > target_affinity_mat.txt 
   

4. setting the hyperparameters in the config_init.py then train and test model under 10fold CV train-test scheme
   Running: 
   + balanced_situatuion on DrugBank 
   * python main.py --root_path "/home/jiani.ma/MULGA/dataset 
                  --dataset "DrugBank"
                  --device "cuda:0"
                  --n_splits 10 
                  --drug_sim_file "drug_affinity_mat.xlsx"
                  --target_sim_file "target_affinity_mat.xlsx"
                  --dti_mat "dti_mat.xlsx"
                  --hgcn_dim 2500
                  --dropout 0.3
                  --epoch_num 1000
                  --lr 0.000005
                  --topk 1 
  
   + imbalanced_situatuion on DrugBank 
   * python main.py --root_path "/home/jiani.ma/MULGA/dataset
                   --dataset "DrugBank"
                   --device "cuda:0"
                   --n_splits 10 
                   --drug_sim_file "drug_affinity_mat.xlsx"
                   --target_sim_file "target_affinity_mat.xlsx"
                   --dti_mat "dti_mat.xlsx"
                   --hgcn_dim 2800
                   --dropout 0.3
                   --epoch_num 1000
                   --lr 0.000008
                   -- topk 10 

   + balanced_situatuion on KIBA 
   * python main.py --root_path "/home/jiani.ma/MULGA/dataset 
                   --dataset "KIBA"
                   --device "cuda:1"
                   --n_splits 10 
                   --drug_sim_file "drug_affinity_mat.xlsx"
                   --target_sim_file "target_affinity_mat.xlsx"
                   --dti_mat "dti_mat.xlsx"
                   --hgcn_dim 1800
                   --dropout 0.5
                   --epoch_num 900
                   --lr 0.00005
                   --topk 1 
  
   + imbalanced_situatuion on KIBA 
   * python main.py --root_path "/home/jiani.ma/MULGA/dataset
                   --dataset "KIBA"
                   --device "cuda:1"
                   --n_splits 10 
                   --drug_sim_file "drug_affinity_mat.xlsx"
                   --target_sim_file "target_affinity_mat.xlsx"
                   --dti_mat "dti_mat.xlsx"
                   --hgcn_dim 1800
                   --dropout 0.5
                   --epoch_num 1000
                   --lr 0.000008
                   --topk 10
     
   + balanced_situatuion on Davis 
   * python main.py --root_path "/home/jiani.ma/MULGA/dataset 
                   --dataset "davis"
                   --device "cuda:0"
                   --n_splits 10 
                   --drug_sim_file "drug_affinity_mat.xlsx"
                   --target_sim_file "target_affinity_mat.xlsx"
                   --dti_mat "dti_mat.xlsx"
                   --hgcn_dim 2000
                   --dropout 0.3
                   --epoch_num 1000
                   --lr 0.0001
                   --topk 1 
  
   + imbalanced_situatuion on Davis 
   * python main.py --root_path "/home/jiani.ma/MULGA/dataset
                   --dataset "davis"
                   --device "cuda:0"
                   --n_splits 10 
                   --drug_sim_file "drug_affinity_mat.xlsx"
                   --target_sim_file "target_affinity_mat.xlsx"
                   --dti_mat "dti_mat.xlsx"
                   --hgcn_dim 1800
                   --dropout 0.5
                   --epoch_num 1000
                   --lr 0.000008
                   --topk 10 

     

### Comparison Methods Rerun 
#### DeepDTA 
* python main.py -- root_path "/home/jiani.ma/MULGA/dataset"
                -- dataset "KIBA"
                -- max_drug_len "drugs.xlsx"
                -- max_target_len "targets.xlsx"
                -- drug_file "drugs.xlsx"
                -- target_file "targets.xlsx"
                -- drug_kernel_size 5
                -- target_kernel_size 11
                -- num_filters 6
                -- embedding_size 128
                -- dropout 0.5
                -- batch_size 256
                -- topk 1
                -- lr 0.00001 
                -- drug_sim_file 'drug_affinity_mat.xlsx'
                -- target_sim_file 'target_affinity_mat.xlsx'
                -- dti_file 'dti_mat.xlsx'
                -- drug_encoder_path "/home/jiani.ma/MULGA/DeepDTA/KIBA_encoder/drug_encoder.xlsx"
                -- target_encoder_path "/home/jiani.ma/MULGA/DeepDTA/KIBA_encoder/target_encoder.xlsx"
                -- device "cuda:1"
                -- n_splits 10

+ (topk can be set to 1 or 10, where 1 denotes pos:neg = 1:1 while 10 denotes pos:neg = 1:10) 

#### DeepConvDTI
* python main.py -- root_path "/home/jiani.ma/MULGA/dataset"
                -- dataset "KIBA"
                -- max_drug_len "drugs.xlsx"
                -- max_target_len "targets.xlsx"
                -- drug_file "drugs.xlsx"
                -- target_file "targets.xlsx"
                -- drug_kernel_size 5
                -- target_kernel_size 11
                -- num_filters 3
                -- embedding_size 128
                -- dropout 0.5
                -- batch_size 256
                -- topk 1
                -- lr 0.000001
                -- drug_sim_file 'drug_affinity_mat.xlsx'
                -- target_sim_file 'target_affinity_mat.xlsx'
                -- dti_file 'dti_mat.xlsx'
                -- drug_encoder_path "/home/jiani.ma/MULGA/DeepConvDTI/KIBA_encoder/drug_encoder.xlsx"
                -- target_encoder_path "/home/jiani.ma/MULGA/DeepConvDTI/KIBA_encoder/target_encoder.xlsx"
                -- device "cuda:0"
                -- n_splits 10

#### GraphDTA 
* python main.py -- num_filters 6
                -- epoch_num 100
                -- lr 0.0005
                -- batch_size 512
                -- device "cuda:0"
                -- dropout 0.3
                -- max_seq_len 1000
                -- embedding_dim 128
                -- target_kernel_size 3 
                -- topk 1 
                -- n_splits 10 
                -- data_folder "/home/jiani.ma/MULGA/dataset/KIBA"
                -- drug_sim_path "drug_affinity_mat.xlsx"
                -- target_sim_path "target_affinity_mat.xlsx"   
                -- DTI_path "dti_mat.xlsx"
                -- drug_smiles_path "drugs.xlsx"drugs.xlsx
                -- target_fasta_path "targets.xlsx"

#### HyperAttentionDTI 
* python main.py -- max_drug_len 100 
                -- max_target_len 1000
                -- root_path "/home/jiani.ma/MULGA/dataset"
                -- dataset "DrugBank"
                -- drug_file "drugs.xlsx"
                -- target_file "targets.xlsx"
                -- drug_sim_file 'drug_affinity_mat.xlsx'
                -- target_sim_file 'target_affinity_mat.xlsx'
                -- drug_num 1822
                -- target_num 1447
                -- conv 40
                -- drug_kernel [4,6,8]
                -- target_kernel [4,8,12]
                -- dropout 0.5
                -- batch_size 128 
                -- topk 1
                -- lr 5e-5

#### MLMC
* python main.py -- m 1822
                 -- n 1447 
                 -- alpha 0.5
                  --mu 10 
                  --lamda 10 
                  --delta 0.5 
                  --data_folder "/home/amber/MULGA/dataset/DrugBank/"
                  --XSD_path "dti_mat.xlsx"
                  --XDD_path "drug_affinity_mat.xlsx"
                  --topk 1
                  

#### LRSpNM
* python main.py -- data_folder "/home/amber/MULGA/LRSpNM/data/KIBA/"
                -- drug_sim_path "drug_sim_mat.xlsx"
                -- target_sim_path "target_sim_mat.xlsx"
                -- DTI_path "dti_mat.xlsx"
                -- m 1720 
                -- n 220 
                -- lambda_d 2
                -- lambda_t 2
                -- alpha  1
                -- mu_1 1
                -- topk 1
                -- p  0.8
                -- n_splits 10 
                -- epoch_num 100











           


