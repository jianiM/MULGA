# MULGA
A unified multi-view graph autoencoder-based approach for identifying drug-protein interaction and drug repositioning

![fig1_](https://user-images.githubusercontent.com/87815194/228164057-ead4748a-64c6-482a-8522-ec1e466b7082.png)



# Datasets 

### DrugBank
4 files:  
1822 drugs 
1447 proteins
6871 validated DTIs  

### KIBA
1720  drugs 
220   proteins
22154 validated DTIs

four files: drugs.xlsx     --- drug id and smiles 
            targets.xlsx   --- uniprot id and fasta
            dti_list.xlsx  --- dti mapping list 
            dti_mat.xlsx   --- dti mat 

## Step-by-step running for MULGA 
### 0 Prepare conda enviroment and install Python libraries needed
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
   + Running: 
   python generate_drug_affinity_mat.py > drug_affinity_mat.txt
   python generate_target_affinity_mat.py > target_affinity_mat.txt 
   

4. setting the hyperparameters in the config_init.py then train and test model under 10fold CV train-test scheme
   + Running: 
   balanced_situatuion on DrugBank 
   python main.py --root_path "/home/jiani.ma/MULGA/dataset 
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
  
   imbalanced_situatuion on DrugBank 
   python main.py --root_path "/home/jiani.ma/MULGA/dataset
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

   balanced_situatuion on KIBA 
   python main.py --root_path "/home/jiani.ma/MULGA/dataset 
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
  
   imbalanced_situatuion on KIBA 
   python main.py --root_path "/home/jiani.ma/MULGA/dataset
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



   




