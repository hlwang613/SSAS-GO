from logzero import logger
import pathlib
from torch.utils.data import DataLoader
from evaluation import Method
from network import *
import torch
import numpy as np
from utils import merge_result, test_performance_gnn_inter, build_label_mapping
import argparse
import pickle as pkl
from config import get_config
import os
import warnings
from graph_data import GoTermDataset, collate_fn
warnings.filterwarnings("ignore")
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def test(config, task, model_pt):  
    print(config.device)
    test_type = getattr(config, 'test_type', 'test')  # 'test' or 'AF2test'
    
    test_set = GoTermDataset(test_type, task) 
    
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    if test_type != 'AF2test':
        annot_path = config.PDB_labels_path
    else:
        annot_path = config.AF2_label_path
        
    labels, idx_goid, goid_idx = build_label_mapping(annot_path, task, expected_dim=test_set.y_true.shape[1], logger=logger )
    labels_num = len(labels)
    pathlib.Path(config.test_result_path).mkdir(exist_ok=True)

    logger.info('Loading Model')
    model = SSAS_Net(graph_size=1280,graph_hid=1280,label_num=labels_num, head=4).to(config.device)
    logger.info(model)
    
    cob_pred_df = []
    k = int(getattr(config, 'top_k', 3))
    for i_t_min in range(k):
        model_path = 'save_models/'+model_pt+'_{0}of{1}.pt'.format(i_t_min, k)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            pred_df = test_performance_gnn_inter(
                model, test_loader, test_set.pdbch_list, idx_goid, goid_idx, task, config.device, 
                save=True, save_file='results/'+model_pt+'_{0}of{1}.pt'.format(i_t_min, k), evaluate=False)
            cob_pred_df.append(pred_df) 
            print(i_t_min, 'epoch:', checkpoint['epoch'], pred_df.shape)
    final_result = merge_result(cob_pred_df)
    with open('results/{}_final.pkl'.format(model_pt), 'wb') as fw:
        pkl.dump(final_result, fw)
    logger.info("Done")
    eval_df = final_result
    if getattr(config, 'test_type', '') == 'AF2test':
        before = len(eval_df)
        eval_df = eval_df[eval_df['gos'].map(lambda g: len(g) > 0)]
        logger.info(f"AF2test: using {len(eval_df)}/{before} proteins with at least one true GO term for evaluation.")
    eval_df = eval_df.reset_index(drop=True)
    pos_count = 0
    try:
        if 'gos' in eval_df.columns:
            pos_count = int(sum(len(g) for g in eval_df['gos']))
    except Exception:
        pos_count = 0
    if pos_count == 0:
        logger.warning('No ground-truth GO annotations found for this test set; skipping metric computation. Predictions saved to results.')
    else:
        y_true = np.zeros((len(eval_df), labels_num), dtype=int)
        y_pred = np.zeros((len(eval_df), labels_num), dtype=float)
        for i, row in eval_df.iterrows():
            for go in row['gos']:
                if go in goid_idx:
                    y_true[i, goid_idx[go]] = 1
            for go, score in row['predictions'].items():
                if go in goid_idx:
                    y_pred[i, goid_idx[go]] = score
        annot_dict = {
            'Y_true': y_true,
            'Y_pred': y_pred,
            'goterms': labels,
            'gonames': None,
            'proteins': eval_df['protein_id'].tolist(),
            'ontology': task
        }
        m = Method(method_name='test', annot_dict=annot_dict)
        fscore, _, _, thresholds = m._protein_centric_fmax() 
        if len(fscore) > 0:
            best_idx = int(np.argmax(fscore))
            fmax = float(fscore[best_idx])
            cut_t = float(thresholds[best_idx])   
        else:
            fmax = 0.0
            cut_t = 1.0
        macro_aupr = float(m.macro_aupr())
        micro_aupr = float(m.micro_aupr())
        protein_aupr = float(m.protein_centric_aupr())
        smin = float(m.smin())
        logger.info(
            f'Evaluation finished: CAFA_Fmax={fmax:.4f}, MacroAUPR={macro_aupr:.4f}, '
            f'MicroAUPR={micro_aupr:.4f}, ProteinAUPR={protein_aupr:.4f}, '
            f'cut-off={cut_t:.2f}, smin={smin:.4f}'
        )
def str2bool(v):
    if isinstance(v,bool):
        return v
    if v == 'True' or v == 'true':
        return True
    if v == 'False' or v == 'false':
        return False

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--task', type=str, default='bp', choices=['bp','mf','cc'], help='')
    p.add_argument('--device', type=str, default='0', help='')
    p.add_argument('--model', type=str, default='', help='')
    p.add_argument('--AF2test', default=False, type=str2bool, help='')
    p.add_argument('--test_type', type=str, default='test', choices=['test','AF2test'], help='')
    p.add_argument('--evaluation', default=True, type=str2bool, help='')
    
    args = p.parse_args()
    print(args)
    config = get_config()
    config.batch_size = 32 
    if args.device != '':
        config.device = "cuda:" + args.device
    config.evaluate = args.evaluation
    if not hasattr(config, 'test_result_path'):
        config.test_result_path = 'results/'
    config.test_type = args.test_type
    test(config, args.task, args.model)
    

