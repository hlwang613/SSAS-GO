
import torch
import torch.nn as nn
import random
import numpy as np 
from torch_geometric.data import Data
import csv
import numpy as np
from joblib import Parallel, delayed, cpu_count
from datetime import datetime
from tqdm import tqdm
import warnings
from sklearn import metrics
from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import dgl
import torch.nn.functional as F
import pickle as pkl
from logzero import logger
from collections import deque
from AverageMeter import AverageMeter
from evaluation import Method
from config import get_config
RES2ID = {
    'A':0,
    'R':1,
    'N':2,
    'D':3,
    'C':4,
    'Q':5,
    'E':6,
    'G':7,
    'H':8,
    'I':9,
    'L':10,
    'K':11,
    'M':12,
    'F':13,
    'P':14,
    'S':15,
    'T':16,
    'W':17,
    'Y':18,
    'V':19,
    '-':20
}


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()

        if self.alpha is not None:
            alpha_t = self.alpha[target]
            focal_loss = alpha_t * focal_loss

        return focal_loss


def merge_result(cob_df_list):
    save_dict = {}
    save_dict['protein_id'] = []
    save_dict['gos'] = []
    save_dict['predictions'] = []
    
    #cob_df_list[0]: first model's prediction dataframe
    for idx, row in cob_df_list[0].iterrows():
        save_dict['protein_id'].append(row['protein_id'])
        save_dict['gos'].append(row['gos'])
        pred_gos = {}
        # merge
        for go, score in row['predictions'].items():
            pred_gos[go] = score
        for single_df in cob_df_list[1:]:
            pred_scores = single_df[single_df['protein_id']==row['protein_id']].reset_index().loc[0, 'predictions']
            for go, score in pred_scores.items():
                pred_gos[go] += score
        # average
        avg_pred_gos = {}
        for go, score in pred_gos.items():
            avg_pred_gos[go] = score/len(cob_df_list)
        
        save_dict['predictions'].append(avg_pred_gos)
        
    df = pd.DataFrame(save_dict)
    
    return df

def pyg_batch_to_dgl(pyg_batch):
    data_list = pyg_batch.to_data_list()
    graphs = []
    for data in data_list:
        g = dgl.graph((data.edge_index[0], data.edge_index[1]), num_nodes=data.x.size(0))
        # Ensure no 0-in-degree nodes for GraphConv
        g = dgl.add_self_loop(g)
        g.ndata['x'] = data.x.float()
        graphs.append(g)
    return dgl.batch(graphs)

def seed_everything(seed: int):
    """
    Set all relevant random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
class AsymmetricLoss(nn.Module):
    
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip  

    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)

        
        pos_loss = target * torch.log(pred_sigmoid.clamp(min=1e-8))
        pos_loss = pos_loss * ((1 - pred_sigmoid) ** self.gamma_pos)

        pred_sigmoid_neg = (pred_sigmoid - self.clip).clamp(min=0)  
        neg_loss = (1 - target) * torch.log((1 - pred_sigmoid_neg).clamp(min=1e-8))
        neg_loss = neg_loss * (pred_sigmoid_neg ** self.gamma_neg)

        loss = -pos_loss - neg_loss
        return loss.mean()


def test_performance_gnn_inter(model, dataloader, test_pid_list, idx_goid, goid_idx, ont, device,
                               save=False, save_file=None, evaluate=True, 
                               with_relations=True, 
                               progress: bool = False, desc: str = None, 
                               pos_weights: torch.Tensor = None,loss_fn=None,
                               go_file: str = get_config().processed_data_dir + "/data/go-basic.obo"):
    model.eval()
    
    true_labels = []
    pred_labels = []
    save_dict = {}
    save_dict['protein_id'] = []
    save_dict['gos'] = []
    save_dict['predictions'] = []

   
    if loss_fn is not None:
        pass 
    elif pos_weights is not None:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    else:
        loss_fn = FocalLoss()
    test_loss_vals = AverageMeter()

    offset = 0
    iterable = dataloader
    if progress:
        try:
            from tqdm import tqdm
            iterable = tqdm(dataloader, desc=desc or 'Valid', leave=False)
        except Exception:
            pass
    for batch in iterable:
        x_test, y_test = batch
        x_test = pyg_batch_to_dgl(x_test).to(device)
        y_test = y_test.to(device).float()
        feats = x_test.ndata['x']

        y_pred = model(x_test, feats)
        loss = loss_fn(y_pred, y_test)
        test_loss_vals.update(loss.item(), len(y_test))

        y_pred = torch.sigmoid(y_pred).detach().cpu().numpy()
        y_test = y_test.detach().cpu().numpy()

        pred_labels.append(y_pred)
        true_labels.append(y_test)

        for rowid in range(y_pred.shape[0]):
            pid = test_pid_list[offset + rowid]
            save_dict['protein_id'].append(pid)
            
            true_gos = set()
            for goidx, goval in enumerate(y_test[rowid]):
                if goval==1:
                    true_gos.add(idx_goid[goidx])
            save_dict['gos'].append(true_gos)
            
            pred_gos = {}
            for goidx, goval in enumerate(y_pred[rowid]):
                pred_gos[idx_goid[goidx]] = goval
            save_dict['predictions'].append(pred_gos)
        offset += y_pred.shape[0]

    true_labels = np.vstack(true_labels)
    pred_labels = np.vstack(pred_labels)

    df = pd.DataFrame(save_dict)
    
    if save and save_file is not None:
        with open(save_file, 'wb') as fw:
            pkl.dump(df, fw)
    if evaluate:
        annot_dict = {
            'Y_true': true_labels,
            'Y_pred': pred_labels,
            'goterms': [idx_goid[i] for i in range(len(idx_goid))],
            'gonames': None,
            'proteins': test_pid_list,
            'ontology': ont
        }
        m = Method(method_name='eval', annot_dict=annot_dict)
        fscore, _, _, thresholds = m._protein_centric_fmax()
        if len(fscore) > 0:
            best_idx = int(np.argmax(fscore))
            best_t = float(thresholds[best_idx])
            best_fmax = float(fscore[best_idx])
        else:
            best_t = 1.0
            best_fmax = 0.0
        macro_aupr = float(m.macro_aupr())
        smin = float(m.smin())
        return best_fmax, macro_aupr, best_t, df, test_loss_vals.avg, smin
    else:
        return df
    
def aa2idx(seq):
   
    # convert letters into numbers
    abc = np.array(list("ARNDCQEGHILKMFPSTWYVX"), dtype='|S1').view(np.uint8)
    idx = np.array(list(seq), dtype='|S1').view(np.uint8)
    for i in range(abc.shape[0]):
        idx[idx == abc[i]] = i

    # treat all unknown characters as gaps
    idx[idx > 20] = 20
    return idx

def protein_graph(sequence, edge_index, esm_embed):
    
    seq_code = aa2idx(sequence)
    seq_code = torch.IntTensor(seq_code)
    # add edge to pairs whose distances are more possible under 8.25
    #row, col = edge_index
    edge_index = torch.LongTensor(edge_index)

    data = Data(x=torch.from_numpy(esm_embed), edge_index=edge_index, native_x=seq_code)
    return data

def pmap_multi(pickleable_fn, data, n_jobs=None, verbose=1, desc=None, **kwargs):
  if n_jobs is None:
    n_jobs = cpu_count() - 1

  results = Parallel(n_jobs=n_jobs, verbose=verbose, timeout=None)(
    delayed(pickleable_fn)(*d, **kwargs) for i, d in tqdm(enumerate(data),desc=desc)
  )

  return results

def pmap_single(pickleable_fn, data, n_jobs=None, verbose=1, desc=None, **kwargs):

    if n_jobs is None:
        n_jobs = cpu_count() - 1

    results = Parallel(n_jobs=n_jobs, verbose=verbose, timeout=None)(
    delayed(pickleable_fn)(d, **kwargs) for i, d in tqdm(enumerate(data),desc=desc)
    )

    return results

def load_predicted_PDB(pdbfile):
    
    # Generate (diagonalized) C_alpha distance matrix from a pdbfile
    parser = PDBParser()
    structure = parser.get_structure(pdbfile.split('/')[-1].split('.')[0], pdbfile)
    residues = [r for r in structure.get_residues()]

    # sequence from atom lines
    records = SeqIO.parse(pdbfile, 'pdb-atom')
    seqs = [str(r.seq) for r in records]

    distances = np.empty((len(residues), len(residues)))
    for x in range(len(residues)):
        for y in range(len(residues)):
            one = residues[x]["CA"].get_coord()
            two = residues[y]["CA"].get_coord()
            distances[x, y] = np.linalg.norm(one-two)

    return distances, seqs[0]


def load_FASTA(filename):
    
    # Loads fasta file and returns a list of the Bio SeqIO records
    infile = open(filename, 'rU')
    entries = []
    proteins = []
    for entry in SeqIO.parse(infile, 'fasta'):
        entries.append(str(entry.seq))
        proteins.append(str(entry.id))
    return proteins, entries


def load_GO_annot(filename):

    # Load GO annotations
    onts = ['mf', 'bp', 'cc']
    prot2annot = {}
    goterms = {ont: [] for ont in onts}
    gonames = {ont: [] for ont in onts}
    with open(filename, mode='r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')

        # molecular function
        next(reader, None)  # skip the headers
        goterms[onts[0]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[0]] = next(reader)

        # biological process
        next(reader, None)  # skip the headers
        goterms[onts[1]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[1]] = next(reader)

        # cellular component
        next(reader, None)  # skip the headers
        goterms[onts[2]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[2]] = next(reader)

        next(reader, None)  # skip the headers
        counts = {ont: np.zeros(len(goterms[ont]), dtype=float) for ont in onts}
        for row in reader:
            prot, prot_goterms = row[0], row[1:]
            prot2annot[prot] = {ont: [] for ont in onts}
            for i in range(3):
                goterm_indices = [goterms[onts[i]].index(goterm) for goterm in prot_goterms[i].split(',') if goterm != '']
                prot2annot[prot][onts[i]] = np.zeros(len(goterms[onts[i]]))
                prot2annot[prot][onts[i]][goterm_indices] = 1.0
                counts[onts[i]][goterm_indices] += 1.0
    return prot2annot, goterms, gonames, counts


def log(*args):
    
    print(f'[{datetime.now()}]', *args)

def PR_metrics(y_true, y_pred):
    
    precision_list = []
    recall_list = []
    threshold = np.arange(0.01,1.01,0.01)
    for T in threshold:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            precision_list.append(metrics.precision_score(y_true, np.where(y_pred>=T, 1, 0)))
            recall_list.append(metrics.recall_score(y_true, np.where(y_pred>=T, 1, 0)))
    return np.array(precision_list), np.array(recall_list)

def fmax(Ytrue, Ypred, nrThresholds):
    
    thresholds = np.linspace(0.0, 1.0, nrThresholds)
    ff = np.zeros(thresholds.shape)
    pr = np.zeros(thresholds.shape)
    rc = np.zeros(thresholds.shape)

    for i, t in enumerate(thresholds):
        thr = np.round(t, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pr[i], rc[i], ff[i], _ = precision_recall_fscore_support(Ytrue, (Ypred >=t).astype(int), average='samples')

    return np.max(ff)

def build_label_mapping(annot_path, task, expected_dim=None, logger=None):
    """
    Load GO terms for a task and build idx<->goid mappings.
    Optionally warn if the loaded label dimension mismatches the expected value.
    """
    _, goterms, _, _ = load_GO_annot(annot_path)
    labels = goterms[task]
    labels_num = len(labels)

    if expected_dim is not None and labels_num != int(expected_dim):
        warn_msg = (
            f"Dataset label_dim ({int(expected_dim)}) != goterms length ({labels_num}); "
            "ensure dataset labels align with annotation order."
        )
        if logger is not None:
            logger.warning(warn_msg)
        else:
            print(warn_msg)

    idx_goid = {idx: go for idx, go in enumerate(labels)}
    goid_idx = {go: idx for idx, go in idx_goid.items()}

    return labels, idx_goid, goid_idx