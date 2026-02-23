from logzero import logger
import dgl
import copy
from AverageMeter import AverageMeter
from torch.utils.data import DataLoader
import torch
from network import *
from utils import *
import argparse
from config import get_config
import numpy as np
from tqdm import tqdm
from graph_data import GoTermDataset, collate_fn 
from utils import test_performance_gnn_inter, pyg_batch_to_dgl
import warnings
import os
warnings.filterwarnings("ignore")


def train(config, task, suffix):
    train_set = GoTermDataset("train", task, config.AF2model)
    pos_weights = train_set.pos_weights.to(config.device).float() 
    valid_set = GoTermDataset("val", task, config.AF2model)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    
    if not config.AF2model:  
        annot_path = config.PDB_labels_path
    else:  
        annot_path = config.AF2_label_path
    labels, idx_goid, goid_idx = build_label_mapping(annot_path, task, expected_dim=train_set.y_true.shape[1], logger=logger)
    labels_num = len(labels)
    
    logger.info(f"Task={task}: dataset label_dim={labels_num}, PDB goterms_len={len(labels)}")
    logger.info('Loading Model')
     
    model = SSAS_Net(graph_size=1280, graph_hid=1280, label_num=labels_num, head=4).to(config.device)
    
    logger.info(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    loss_type = getattr(config, 'loss_type', 'focal')
    if loss_type == 'asl':
        loss_fn = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
        logger.info("Using AsymmetricLoss (gamma_neg=4, gamma_pos=1, clip=0.05)")
    elif loss_type == 'focal':
        loss_fn = FocalLoss()
        logger.info("Using FocalLoss (gamma=2)")
    elif loss_type == 'bce':
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)  
        logger.info("Using BCEWithLogitsLoss")
            
    best_models = []
    k = int(getattr(config, 'top_k', 3))
    metric_name = getattr(config, 'selection_metric', 'fmax').lower()
    
    for e in range(config.max_epochs):
        model.train()
        train_loss_vals = AverageMeter()
        with tqdm(total=len(train_loader), desc=f"Epoch {e+1}/{config.max_epochs}", leave=True) as pbar:
            for pyg_batch, labels in train_loader:
                batched_graph = pyg_batch_to_dgl(pyg_batch).to(config.device)
                labels = labels.to(config.device).float()
                feats = batched_graph.ndata['x']

                logits = model(batched_graph, feats)
                loss = loss_fn(logits, labels)
                
                train_loss_vals.update(loss.item(), len(labels))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix(train_loss=f"{train_loss_vals.val:.6f}", avg=f"{train_loss_vals.avg:.6f}")
                pbar.update(1)
        
        model.eval()
        
        plus_fmax, plus_aupr, plus_t, df, valid_loss_avg, _ = test_performance_gnn_inter(
            model,
            val_loader,
            valid_set.pdbch_list,
            idx_goid, goid_idx, 
            task, config.device, 
            progress=True, 
            desc=f"Valid {e+1}/{config.max_epochs}",
            loss_fn=loss_fn
            )
        logger.info('Epoch: {}, Train Loss: {:.6f}\tValid Loss: {:.6f}, CAFA_Fmax: {:.4f}, MacroAUPR: {:.4f}, cut-off: {:.2f}, df_shape: {}'.format(
            e, train_loss_vals.avg, valid_loss_avg, plus_fmax, plus_aupr, plus_t, df.shape))
        
        current_metric = float(plus_fmax if metric_name == 'fmax' else plus_aupr)

        if len(best_models) < k:
            slot = len(best_models)
            best_models.append({"metric": current_metric, "epoch": e, "slot": slot})
            torch.save({'epoch': e,'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, 
                    './save_models/model_{0}_{1}_{2}of{3}.pt'.format(task, suffix, slot, k))
            logger.info(f"Saved checkpoint to slot {slot} (metric={current_metric:.4f})")
        else:
            worst_idx = int(np.argmin([bm["metric"] for bm in best_models]))
            if current_metric > best_models[worst_idx]["metric"]:
                slot = best_models[worst_idx]["slot"]
                best_models[worst_idx] = {"metric": current_metric, "epoch": e, "slot": slot}
                torch.save({'epoch': e,'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}, 
                        './save_models/model_{0}_{1}_{2}of{3}.pt'.format(task, suffix, slot, k))
                logger.info(f"Replaced slot {slot} with better checkpoint (metric={current_metric:.4f})")
    logger.info("Training/validation completed. Saved checkpoints under ./save_models")
    if best_models:
        best_entry = max(best_models, key=lambda bm: bm["metric"])
        best_path = './save_models/model_{0}_{1}_{2}of{3}.pt'.format(task, suffix, best_entry["slot"], k)
        logger.info(f"Training/validation completed. Best slot={best_entry['slot']} metric={best_entry['metric']:.4f}. Saved under ./save_models")
        return best_path, best_entry["metric"]
    else:
        logger.info("Training/validation completed but no checkpoints recorded.")
        return None, None

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
    p.add_argument('--suffix', type=str, default='CLAF', help='')
    p.add_argument('--device', type=str, default='0', help='cpu or cuda:0')
    p.add_argument('--AF2model', default=False, type=str2bool, help='whether to use AF2model for training')
    p.add_argument('--batch_size', type=int, default=64, help='')
    # p.add_argument('--seeds', type=str, default='', help='例如: 42,1337,2024')

    args = p.parse_args()
    config = get_config()
    config.optimizer['lr'] = 1e-4
    config.batch_size = args.batch_size
    config.max_epochs = 12
    if args.device != '':
        config.device = "cuda:" + args.device
    print(args)
    config.AF2model = args.AF2model
    train(config, args.task, args.suffix)
