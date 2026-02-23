import ml_collections
import copy

CONFIG = ml_collections.ConfigDict({
    "debug": False,
    "max_epochs": 12,
    "batch_size": 64,
    "device": 'cuda:0',
    "save_models_dir": "./save_models",
    "results_dir": "./results",
    "PDB_labels_path": "/root/autodl-tmp/data/nrPDB-GO_2019.06.18_annot.tsv",
    "AF2_label_path": "/root/autodl-tmp/data/nrSwiss-Model-GO_annot.tsv",
    "processed_data_dir": "/root/autodl-tmp/",
    "AF2model": False,
    "optimizer":{
        "lr": 1e-4,
        "betas": (0.95, 0.95),
        "eps": 1e-8,
    },
    
    "scheduler":{
        "step_size": 300,
        "gamma": 0.75,
    },
    # 'focal' | 'asl' | 'bce'
    "loss_type": "bce",  
    
    "top_k": 3,
    "selection_metric": "fmax",  # fmax | aupr
})
def get_config():
    config = copy.deepcopy(CONFIG)
    return config
