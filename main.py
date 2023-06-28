import torch
import numpy as np
import os
import random
import argparse
from exp.exp import Exp
from logger import create_logger


seed = 3047
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser(description='DeepRicci')

# Experiment settings
parser.add_argument('--downstream_task', type=str, default='classification',
                    choices=['classification', 'clustering'])
parser.add_argument('--dataset', type=str, default='Cora',
                    choices=['Cora', 'Citeseer', 'chameleon', 'squirrel'])
parser.add_argument('--is_graph', type=bool, default=True)
parser.add_argument('--root_path', type=str, default='./datasets')
parser.add_argument('--eval_freq', type=int, default=50)
parser.add_argument('--exp_iters', type=int, default=5)
parser.add_argument('--version', type=str, default="run")
parser.add_argument('--log_path', type=str, default="./results/v2302152230/cls_Cora.log")

# Deep Ricci Graph Contrastive Learning Module
parser.add_argument('--backbone', type=str, default='spgat', choices=['gcn', 'spgat', 'gat', 'sage'])
parser.add_argument('--backbone_topk', type=int, default=8)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--hidden_features', type=int, default=512)
parser.add_argument('--embed_features', type=int, default=32, help='dimensions of graph embedding')
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--dropout_node', type=float, default=0.5)
parser.add_argument('--dropout_edge', type=float, default=0.75)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--w_decay', type=float, default=0.)
parser.add_argument('--update_adj_ratio', type=float, default=0.1)
parser.add_argument('--act_adj', type=str, default='elu', choices=['relu', 'elu', 'gumbel'])
parser.add_argument('--act_func', type=str, default='tanh', choices=['relu', 'tanh'])
parser.add_argument('--wq_type', type=str, default='mlp', choices=['linear', 'mlp'])
parser.add_argument('--n_heads', type=int, default=8, help='number of attention heads')
parser.add_argument('--d_riemann', type=int, default=32, help='dimension of Riemannian embedding')
parser.add_argument('--d_hyla', type=int, default=256, help='dimension of Riemannian embedding')
parser.add_argument('--scale', type=float, default=0.1, help='scale for sampling eigenvalues')
parser.add_argument('--cur_h', type=float, default=-1., help='curvature of hyperbolic')
parser.add_argument('--cur_s', type=float, default=1., help='curvature of sphere')
parser.add_argument('--s', type=float, default=1., help='for Fermi-Dirac decoder')
parser.add_argument('--r', type=float, default=2., help='Fermi-Dirac decoder')
parser.add_argument('--gamma', type=float, default=5., help='coefficient for structural loss')
parser.add_argument('--eps', type=float, default=None, help='threshold')
parser.add_argument('--topk', type=int, default=60, help='select topk numbers')
parser.add_argument('--temperature', type=float, default=0.2, help='temperature of NT-Xent loss')

# Evaluation NetWork for Classification
parser.add_argument('--alpha_gat', type=float, default=0.2)
parser.add_argument('--n_heads_gat', type=int, default=8, help='number of attention heads of gat')
parser.add_argument('--hidden_features_cls', type=int, default=8)
parser.add_argument('--dropout_node_cls', type=float, default=0.6)
parser.add_argument('--dropout_edge_cls', type=float, default=0.6)
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999))
parser.add_argument('--n_layers_cls', type=int, default=2)
parser.add_argument('--lr_cls', type=float, default=0.005)
parser.add_argument('--w_decay_cls', type=float, default=5e-4)
parser.add_argument('--epochs_cls', type=int, default=200)
parser.add_argument('--patience_cls', type=int, default=10)
parser.add_argument('--save_path_cls', type=str, default='./checkpoints/cls.pth')

# Evaluation NetWork for Clustering
parser.add_argument('--n_cluster_trials', type=int, default=5)

# Structure Bootstrapping
parser.add_argument('--tau', type=float, default=0.9999)
parser.add_argument('--iterations', type=int, default=0)

# GPU
parser.add_argument('--use_gpu', action='store_false', help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')

configs = parser.parse_args()
log_path = f"./results/{configs.version}/{configs.downstream_task}_{configs.backbone}_{configs.dataset}.log"
configs.log_path = log_path
if not os.path.exists(f"./results"):
    os.mkdir("./results")
if not os.path.exists(f"./results/{configs.version}"):
    os.mkdir(f"./results/{configs.version}")
print(f"Log path: {configs.log_path}")
logger = create_logger(configs.log_path)
logger.info(configs)

exp = Exp(configs)
exp.train()
torch.cuda.empty_cache()

