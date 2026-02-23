import pandas as pd
import torch
from logzero import logger
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
import numpy as np
import os
from config import get_config
from utils import load_GO_annot

def collate_fn(batch):
    
    graphs, y_trues = map(list, zip(*batch))
    return Batch.from_data_list(graphs), torch.stack(y_trues).float()

class GoTermDataset(Dataset):

    def __init__(self, set_type, task, AF2model=False,homology_threshold=None):
        
       
        self.task = task
       
        if set_type != 'AF2test':
           
            prot2annot, goterms, gonames, counts = load_GO_annot(get_config().PDB_labels_path)
        else:
           
            prot2annot, goterms, gonames, counts = load_GO_annot(get_config().AF2_label_path)
        goterms = goterms[self.task]
        gonames = gonames[self.task]
        
        output_dim = len(goterms)
        class_sizes = counts[self.task]
        mean_class_size = np.mean(class_sizes)
        pos_weights = mean_class_size / class_sizes
        pos_weights = np.maximum(1.0, np.minimum(10.0, pos_weights))
        
        self.pos_weights = torch.tensor(pos_weights).float()

        self.processed_dir = get_config().processed_data_dir

        self.graph_list = torch.load(os.path.join(self.processed_dir, f"{set_type}_graph.pt"))
        
        if set_type == 'AF2test':
            self.pdbch_list = torch.load(os.path.join(self.processed_dir, f"{set_type}_pdbch.pt"))["test_pdbch"]
        else:
            self.pdbch_list = torch.load(os.path.join(self.processed_dir, f"{set_type}_pdbch.pt"))[f"{set_type}_pdbch"]
        self.y_true = np.stack([prot2annot[pdb_c][self.task] for pdb_c in self.pdbch_list])
        self.y_true = torch.tensor(self.y_true)

        if AF2model:
            prot2annot, goterms, gonames, counts = load_GO_annot(get_config().AF2_label_path)
            
            graph_list_af = torch.load(os.path.join(self.processed_dir, f"AF2{set_type}_graph.pt"))
            
            self.graph_list += graph_list_af
            self.pdbch_list_af = torch.load(os.path.join(self.processed_dir, f"AF2{set_type}_pdbch.pt"))[f"{set_type}_pdbch"]
            y_true_af = np.stack([prot2annot[pdb_c][self.task] for pdb_c in self.pdbch_list_af])
            
            
            self.y_true = np.concatenate([self.y_true, y_true_af],0)
            self.y_true = torch.tensor(self.y_true)

            try:
                self.pdbch_list = list(self.pdbch_list) + list(self.pdbch_list_af)
            except Exception:
                self.pdbch_list = list(self.pdbch_list)
                self.pdbch_list_af = list(self.pdbch_list_af)
                self.pdbch_list = self.pdbch_list + self.pdbch_list_af
        
        try:
            n_graphs = len(self.graph_list)
            n_labels = int(self.y_true.shape[1]) if hasattr(self.y_true, 'shape') else None
            logger.info(f"Loaded dataset split={set_type} task={self.task} AF2model={AF2model}: graphs={n_graphs}, labels_dim={n_labels}")
        except Exception:
            pass

    def __getitem__(self, idx):
        
        
        return self.graph_list[idx], self.y_true[idx]

    def __len__(self):
       
        return len(self.graph_list)
