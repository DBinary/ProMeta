import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

def generate_pathway_mask(proteins_list, cpdb_path):
    """
    Generates a binary mask mapping proteins to pathways using CPDB data.
    """
    print(f"[Info] Generating pathway mask from {cpdb_path}...")
    
    # 1. Clean Protein IDs
    protein2id = {}
    for i, p in enumerate(proteins_list):
        clean_name = str(p).strip().upper().split('.')[0]
        protein2id[clean_name] = i
        
    num_proteins = len(proteins_list)
    
    if not os.path.exists(cpdb_path):
        raise FileNotFoundError(f"CPDB file not found at: {cpdb_path}")

    # Load CPDB data
    cpdb = pd.read_csv(cpdb_path, sep='\t', dtype=str)
    
    # Handle column name variations
    gene_col = 'hgnc_symbol_ids' if 'hgnc_symbol_ids' in cpdb.columns else 'hgnc_symbol'
    cpdb = cpdb.dropna(subset=[gene_col])
    cpdb.drop_duplicates(subset=['pathway', 'source'], inplace=True)
    cpdb = cpdb[cpdb['source'] == 'KEGG'] # Optional filter
    
    pathways = cpdb['pathway'].unique()
    local2id = {local: idx for idx, local in enumerate(pathways)}
    num_pathways = len(pathways)
    
    local2gene = np.zeros((num_pathways, num_proteins), dtype=np.float32)
    print(f"[Info] Mapping {num_proteins} proteins to {num_pathways} pathways...")
    
    matched_count = 0
    for local, genes_str in zip(cpdb['pathway'].values, cpdb[gene_col].values):
        genes = genes_str.split(',')
        for gene in genes:
            gene_clean = gene.strip().upper()
            if gene_clean in protein2id:
                local2gene[local2id[local], protein2id[gene_clean]] = 1.0
                matched_count += 1
                
    col_sums = local2gene.sum(axis=0)
    unknown_indices = np.where(col_sums == 0)[0].tolist()
    
    print(f"Mask Shape: {local2gene.shape}")
    print(f"Proteins mapped to pathways: {num_proteins - len(unknown_indices)}")
    print(f"Unknown (Unmapped) Proteins: {len(unknown_indices)}")
    
    return torch.tensor(local2gene), unknown_indices

class MetaDataset(Dataset):
    """
    Custom Dataset for Meta-Learning (Few-Shot).
    """
    def __init__(self, feature_matrix, case_dict, control_dict, eid_to_idx, 
                 support_size=32, max_support_size=32, query_size=128, 
                 mode='train', random_seed=42):
        self.features = feature_matrix
        self.tasks = []
        self.eid_to_idx = eid_to_idx
        self.support_size = support_size
        self.query_size = query_size
        self.mode = mode
        self.seed = random_seed
        
        valid_terms = [t for t in case_dict.keys() if t in control_dict]
        self.term2support = {}
        self.term2query = {}
        
        for i, term in enumerate(valid_terms):
            case_indices = [eid_to_idx[str(e)] for e in case_dict[term] if str(e) in eid_to_idx]
            ctrl_indices = [eid_to_idx[str(e)] for e in control_dict[term] if str(e) in eid_to_idx]
            
            rng = np.random.RandomState(self.seed + i)
            rng.shuffle(case_indices)
            rng.shuffle(ctrl_indices)
            
            try:
                all_support_case = case_indices[:max_support_size//2]
                all_support_ctrl = ctrl_indices[:max_support_size//2]
                
                self.term2support[term] = (
                    all_support_case[:self.support_size//2], 
                    all_support_ctrl[:self.support_size//2]
                )
                
                rem_case = case_indices[max_support_size//2:]
                rem_ctrl = ctrl_indices[max_support_size//2:]
                
                n_case_query = min(len(rem_case), query_size//4)
                n_ctrl_query = min(len(rem_ctrl), query_size - n_case_query)
                
                self.term2query[term] = (rem_case[:n_case_query], rem_ctrl[:n_ctrl_query])
                self.tasks.append(term)
            except Exception as e:
                # print(f"[Warning] Term {term} skipped: {e}")
                continue
            
        print(f"[{mode.upper()}] Prepared {len(self.tasks)} tasks.")

    def __len__(self): return len(self.tasks)
    
    def __getitem__(self, idx):
        term = self.tasks[idx]
        sup_case_idx, sup_ctrl_idx = self.term2support[term]
        qry_case_idx, qry_ctrl_idx = self.term2query[term]

        sX = torch.tensor(self.features[np.concatenate([sup_case_idx, sup_ctrl_idx])], dtype=torch.float32)
        sY = torch.tensor(np.concatenate([np.ones(len(sup_case_idx)), np.zeros(len(sup_ctrl_idx))]), dtype=torch.float32)
        qX = torch.tensor(self.features[np.concatenate([qry_case_idx, qry_ctrl_idx])], dtype=torch.float32)
        qY = torch.tensor(np.concatenate([np.ones(len(qry_case_idx)), np.zeros(len(qry_ctrl_idx))]), dtype=torch.float32)
        
        return qX, qY, sX, sY, term