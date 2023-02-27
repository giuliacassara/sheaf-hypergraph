#!/usr/bin/env python
# coding: utf-8

import os
import time
# import math
import torch
# import pickle
import argparse

import numpy as np
import os.path as osp
import scipy.sparse as sp
import torch_sparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm

from layers import *
from models import *
from edgnn import EquivSetGNN, SheafEquivSetGNN
from preprocessing import *

from convert_datasets_to_pygDataset import dataset_Hypergraph


def parse_method(args, data):
    #     Currently we don't set hyperparameters w.r.t. different dataset
    if args.method == 'AllSetTransformer':
        if args.LearnMask:
            model = SetGNN(args, data.norm)
        else:
            model = SetGNN(args)
    
    elif args.method == 'AllDeepSets':
        args.PMA = False
        args.aggregate = 'add'
        if args.LearnMask:
            model = SetGNN(args,data.norm)
        else:
            model = SetGNN(args)

#     elif args.method == 'SetGPRGNN':
#         model = SetGPRGNN(args)

    elif args.method == 'CEGCN':
        model = CEGCN(in_dim=args.num_features,
                      hid_dim=args.MLP_hidden,  # Use args.enc_hidden to control the number of hidden layers
                      out_dim=args.num_classes,
                      num_layers=args.All_num_layers,
                      dropout=args.dropout,
                      Normalization=args.normalization)

    elif args.method == 'CEGAT':
        model = CEGAT(in_dim=args.num_features,
                      hid_dim=args.MLP_hidden,  # Use args.enc_hidden to control the number of hidden layers
                      out_dim=args.num_classes,
                      num_layers=args.All_num_layers,
                      heads=args.heads,
                      output_heads=args.output_heads,
                      dropout=args.dropout,
                      Normalization=args.normalization)

    elif args.method == 'HyperGCN':
        #         ipdb.set_trace()
        He_dict = get_HyperGCN_He_dict(data)
        model = HyperGCN(V=data.x.shape[0],
                         E=He_dict,
                         X=data.x,
                         num_features=args.num_features,
                         num_layers=args.All_num_layers,
                         num_classses=args.num_classes,
                         args=args
                         )
    elif args.method == 'SheafHyperGCNDiag':
        #         ipdb.set_trace()
        He_dict = get_HyperGCN_He_dict(data)
        model = SheafHyperGCN(V=data.x.shape[0],
                         num_features=args.num_features,
                         num_layers=args.All_num_layers,
                         num_classses=args.num_classes,
                         args=args, sheaf_type= 'DiagSheafs'
                         )
    elif args.method == 'SheafHyperGCNOrtho':
        #         ipdb.set_trace()
        He_dict = get_HyperGCN_He_dict(data)
        model = SheafHyperGCN(V=data.x.shape[0],
                         num_features=args.num_features,
                         num_layers=args.All_num_layers,
                         num_classses=args.num_classes,
                         args=args, sheaf_type= 'OrthoSheafs'
                         )
    elif args.method == 'SheafHyperGCNGeneral':
        #         ipdb.set_trace()
        He_dict = get_HyperGCN_He_dict(data)
        model = SheafHyperGCN(V=data.x.shape[0],
                         num_features=args.num_features,
                         num_layers=args.All_num_layers,
                         num_classses=args.num_classes,
                         args=args, sheaf_type= 'GeneralSheafs'
                         )
    elif args.method == 'SheafHyperGCNLowRank':
        #         ipdb.set_trace()
        model = SheafHyperGCN(V=data.x.shape[0],
                         num_features=args.num_features,
                         num_layers=args.All_num_layers,
                         num_classses=args.num_classes,
                         args=args, sheaf_type= 'LowRankSheafs'
                         )

    elif args.method == 'HGNN':
        # model = HGNN(in_ch=args.num_features,
        #              n_class=args.num_classes,
        #              n_hid=args.MLP_hidden,
        #              dropout=args.dropout)
        model = HCHA(args)

    elif args.method == 'HNHN':
        model = HNHN(args)

    elif args.method == 'HCHA':
        model = HCHA(args)

    elif args.method == 'MLP':
        model = MLP_model(args)

    elif args.method in ['DiagSheafs', 'OrthoSheafs', 'GeneralSheafs', 'LowRankSheafs', 'DiagSheafsDiffusion', 'OrthoSheafsDiffusion', 'GeneralSheafsDiffusion', 'LowRankSheafsDiffusion']:
        model = HyperSheafs(args, args.method)


    # elif args.method == 'OrthoSheafs':
    #     model = OrthoSheafs(args)
    # elif args.method == 'GeneralSheafs':
    #     model = GeneralSheafs(args)
    elif args.method == 'EquivSetGNN':
        model = EquivSetGNN(num_features=args.num_features, num_classes=args.num_classes, args=args)
    elif args.method == 'SheafEquivSetGNN_Diag':
        model = SheafEquivSetGNN(num_features=args.num_features, num_classes=args.num_classes, sheaf_type='DiagEDGNN', args=args)
    elif args.method == 'SheafEquivSetGNN_Ortho':
        model = SheafEquivSetGNN(num_features=args.num_features, num_classes=args.num_classes, sheaf_type='OrthoEDGNN', args=args)
    elif args.method == 'SheafEquivSetGNN_General':
        model = SheafEquivSetGNN(num_features=args.num_features, num_classes=args.num_classes, sheaf_type='GeneralEDGNN', args=args)
    elif args.method == 'SheafEquivSetGNN_LowRank':
        model = SheafEquivSetGNN(num_features=args.num_features, num_classes=args.num_classes, sheaf_type='LowRankEDGNN', args=args)
    return model


class Logger(object):
    """ Adapted from https://github.com/snap-stanford/ogb/ """

    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            return best_result[:, 1], best_result[:, 3]

    def plot_result(self, run=None):
        plt.style.use('seaborn')
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            x = torch.arange(result.shape[0])
            plt.figure()
            print(f'Run {run + 1:02d}:')
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(['Train', 'Valid', 'Test'])
        else:
            result = 100 * torch.tensor(self.results[0])
            x = torch.arange(result.shape[0])
            plt.figure()
#             print(f'Run {run + 1:02d}:')
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(['Train', 'Valid', 'Test'])


@torch.no_grad()
def evaluate(model, data, split_idx, eval_func, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        out = model(data)
        out = F.log_softmax(out, dim=1)

    train_acc = eval_func(
        data.y[split_idx['train']], out[split_idx['train']], name='train')
    valid_acc = eval_func(
        data.y[split_idx['valid']], out[split_idx['valid']], name='valid')
    test_acc = eval_func(
        data.y[split_idx['test']], out[split_idx['test']], name='test')

#     Also keep track of losses
    train_loss = F.nll_loss(
        out[split_idx['train']], data.y[split_idx['train']])
    valid_loss = F.nll_loss(
        out[split_idx['valid']], data.y[split_idx['valid']])
    test_loss = F.nll_loss(
        out[split_idx['test']], data.y[split_idx['test']])
    return train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, out


def eval_acc(y_true, y_pred, name):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=False).detach().cpu().numpy()

#     ipdb.set_trace()
#     for i in range(y_true.shape[1]):
    is_labeled = y_true == y_true
    correct = y_true[is_labeled] == y_pred[is_labeled]
    if len(correct) == 0:
        acc_list.append(0.0)
    else:    
        acc_list.append(float(np.sum(correct))/len(correct))
    
    return sum(acc_list)/len(acc_list)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- Main part of the training ---
# # Part 0: Parse arguments


"""

"""

if __name__ == '__main__':
    start_time = time.time()
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_prop', type=float, default=0.5)
    parser.add_argument('--valid_prop', type=float, default=0.25)
    parser.add_argument('--dname', default='walmart-trips-100')
    # method in ['SetGNN','CEGCN','CEGAT','HyperGCN','HGNN','HCHA']
    parser.add_argument('--method', default='AllSetTransformer')
    parser.add_argument('--epochs', default=500, type=int)
    # Number of runs for each split (test fix, only shuffle train/val)
    parser.add_argument('--runs', default=10, type=int)
    parser.add_argument('--cuda', default=0, choices=[-1, 0, 1], type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--wd', default=0.0, type=float)
    # How many layers of full NLConvs
    parser.add_argument('--All_num_layers', default=2, type=int)
    parser.add_argument('--MLP_num_layers', default=2,
                        type=int)  # How many layers of encoder
    parser.add_argument('--MLP_hidden', default=64,
                        type=int)  # Encoder hidden units
    parser.add_argument('--Classifier_num_layers', default=2,
                        type=int)  # How many layers of decoder
    parser.add_argument('--Classifier_hidden', default=64,
                        type=int)  # Decoder hidden units
    parser.add_argument('--display_step', type=int, default=-1)
    parser.add_argument('--aggregate', default='mean', choices=['sum', 'mean'])
    # ['all_one','deg_half_sym']
    parser.add_argument('--normtype', default='all_one')
    parser.add_argument('--add_self_loop', action='store_false')
    # NormLayer for MLP. ['bn','ln','None']
    parser.add_argument('--normalization', default='ln')
    parser.add_argument('--deepset_input_norm', default = True)
    parser.add_argument('--GPR', action='store_false')  # skip all but last dec
    # skip all but last dec
    parser.add_argument('--LearnMask', action='store_false')
    parser.add_argument('--num_features', default=0, type=int)  # Placeholder
    parser.add_argument('--num_classes', default=0, type=int)  # Placeholder
    # Choose std for synthetic feature noise
    parser.add_argument('--feature_noise', default='1', type=str)
    # whether the he contain self node or not
    parser.add_argument('--exclude_self', action='store_true')
    parser.add_argument('--PMA', action='store_true')
    #     Args for HyperGCN
    parser.add_argument('--HyperGCN_mediators', action='store_true')
    parser.add_argument('--HyperGCN_fast', action='store_false')
    #     Args for Attentions: GAT and SetGNN
    parser.add_argument('--heads', default=1, type=int)  # Placeholder
    parser.add_argument('--output_heads', default=1, type=int)  # Placeholder
    #     Args for HNHN
    parser.add_argument('--HNHN_alpha', default=-1.5, type=float)
    parser.add_argument('--HNHN_beta', default=-0.5, type=float)
    parser.add_argument('--HNHN_nonlinear_inbetween', default=True, type=bool)
    #     Args for HCHA
    parser.add_argument('--HCHA_symdegnorm', action='store_true')
    #     Args for UniGNN
    parser.add_argument('--UniGNN_use-norm', action="store_true", help='use norm in the final layer')
    parser.add_argument('--UniGNN_degV', default = 0)
    parser.add_argument('--UniGNN_degE', default = 0)
    parser.add_argument('--wandb', default=True, type=bool)
    parser.add_argument('--activation', default='relu', choices=['Id','relu', 'prelu'])
    
    # Args just for EDGNN
    parser.add_argument('--MLP2_num_layers', default=-1, type=int, help='layer number of mlp2')
    parser.add_argument('--MLP3_num_layers', default=-1, type=int, help='layer number of mlp3')
    parser.add_argument('--edconv_type', default='EquivSet', type=str, choices=['EquivSet', 'JumpLink', 'MeanDeg', 'Attn', 'TwoSets'])
    parser.add_argument('--restart_alpha', default=0.5, type=float)

    # Args for Sheaves
    parser.add_argument('--init_hedge', default="rand", type=str, choices=['rand', 'avg']) 
    parser.add_argument('--use_attention', type=str2bool, default=True) #used in HCHA. if true ypergraph attention otherwise hypergraph conv
    parser.add_argument('--tag', type=str, default='testing') #helper for wandb in order to filter out the testing runs. if set to testing we are in dev mode
    parser.add_argument('--sheaf_normtype', type=str, default='degree_norm', choices=['degree_norm', 'block_norm', 'sym_degree_norm', 'sym_block_norm']) #used to normalise the sheaf laplacian. will add other normalisations later
    parser.add_argument('--sheaf_act', type=str, default='sigmoid', choices=['sigmoid', 'tanh', 'none']) #final activation used after predicting the dxd block
    parser.add_argument('--sheaf_dropout', type=str2bool, default=False) #final activation used after predicting the dxd block
    parser.add_argument('--sheaf_left_proj', type=str2bool, default=False) #multiply to the left with IxW
    parser.add_argument('--dynamic_sheaf', type=str2bool, default=False) #if set to True, a different sheaf is predicted at each layer
    parser.add_argument('--sheaf_special_head', type=str2bool, default=False) #if set to True, a special head corresponding to alpha=1 and d=heads-1 in that case)
    parser.add_argument('--sheaf_pred_block', type=str, default="MLP_var1") #if set to True, a special head corresponding to alpha=1 and d=heads-1 in that case)
    parser.add_argument('--sheaf_transformer_head', type=int, default=1) #only when sheaf_pred_block==transformer. The number of transformer head used to predict the dxd blocks
    parser.add_argument('--AllSet_input_norm', default=True)
    
    parser.add_argument('--residual_HCHA', default=False) # HCHA and *Sheafs only; if HCHA-based architectures have conv layers with residual connections
    
    parser.add_argument('--rank', default=0, type=int, help='rank for dxd blocks in LowRankSheafs') # ronly for ank for the low-rank matrix generation of the dxd block                                                                                          # should be < d
    # parser.add_argument('--noisy_low_rank', default=False) #only for ank for the low-rank . tell us if we want to add some gaussian noise on the diagonal of the low-rank matrices

    parser.set_defaults(PMA=True)  # True: Use PMA. False: Use Deepsets.
    parser.set_defaults(add_self_loop=True)
    parser.set_defaults(exclude_self=False)
    parser.set_defaults(GPR=False)
    parser.set_defaults(LearnMask=False)
    parser.set_defaults(HyperGCN_mediators=True)
    parser.set_defaults(HyperGCN_fast=True)
    parser.set_defaults(HCHA_symdegnorm=False)
    
    #     Use the line below for .py file
    args = parser.parse_args()
    #     Use the line below for notebook
    # args = parser.parse_args([])
    # args, _ = parser.parse_known_args()
    
    
    # # Part 1: Load data
    if args.wandb:
        import wandb
        wandb.init(sync_tensorboard=False, project='hyper_sheaf', reinit = False, config = args, entity='hyper_graphs', tags=[args.tag])
        print('Monitoring using wandb')
    
    ### Load and preprocess data ###
    existing_dataset = ['20newsW100', 'ModelNet40', 'zoo',
                        'NTU2012', 'Mushroom',
                        'coauthor_cora', 'coauthor_dblp',
                        'yelp', 'amazon-reviews', 'walmart-trips', 'house-committees',
                        'walmart-trips-100', 'house-committees-100',
                        'cora', 'citeseer', 'pubmed', 'congress-bills', 'senate-committees', 'senate-committees-100']
        
    # if args.dname in existing_dataset:
    #     dname = args.dname
    #     if dname in ['cora', 'citeseer','pubmed']:
    #             p2raw = '../data/AllSet_all_raw_data/cocitation/'
    #     elif dname in ['coauthor_cora', 'coauthor_dblp']:
    #             p2raw = '../data/AllSet_all_raw_data/coauthorship/'
    #     dataset = dataset_Hypergraph(name=dname,root = '../data/pyg_data/hypergraph_dataset_updated/',
    #                                      p2raw = p2raw)
    #     data = dataset.data
    #     args.num_features = dataset.num_features
    #     args.num_classes = dataset.num_classes
    #         #         Shift the y label to start with 0
    #     if not hasattr(data, 'n_x'):
    #         data.n_x = torch.tensor([data.x.shape[0]])
    #     if not hasattr(data, 'num_hyperedges'):
    #         # note that we assume the he_id is consecutive.
    #         data.num_hyperedges = torch.tensor(
    #             [data.edge_index[0].max()-data.n_x[0]+1])

    synthetic_list = ['amazon-reviews', 'walmart-trips', 'house-committees', 'walmart-trips-100', 'house-committees-100', 'congress-bills', 'senate-committees', 'senate-committees-100']
    
    if args.dname in existing_dataset:
        dname = args.dname
        f_noise = args.feature_noise
        if (f_noise is not None) and dname in synthetic_list:
            p2raw = '../data/AllSet_all_raw_data/'
            dataset = dataset_Hypergraph(name=dname, 
                    feature_noise=f_noise,
                    p2raw = p2raw)
        else:
            if dname in ['cora', 'citeseer','pubmed']:
                p2raw = '../data/AllSet_all_raw_data/cocitation/'
            elif dname in ['coauthor_cora', 'coauthor_dblp']:
                p2raw = '../data/AllSet_all_raw_data/coauthorship/'
            elif dname in ['yelp']:
                p2raw = '../data/AllSet_all_raw_data/yelp/'
            else:
                p2raw = '../data/AllSet_all_raw_data/'
            dataset = dataset_Hypergraph(name=dname,root = '../data/pyg_data/hypergraph_dataset_updated/',
                                         p2raw = p2raw)
        data = dataset.data
        args.num_features = dataset.num_features
        args.num_classes = dataset.num_classes
        if args.dname in ['yelp', 'walmart-trips', 'house-committees', 'walmart-trips-100', 'house-committees-100', 'senate-committees', 'senate-committees-100', 'congress-bills']:
            #         Shift the y label to start with 0
            args.num_classes = len(data.y.unique())
            data.y = data.y - data.y.min()
        if not hasattr(data, 'n_x'):
            data.n_x = torch.tensor([data.x.shape[0]])
        if not hasattr(data, 'num_hyperedges'):
            # note that we assume the he_id is consecutive.
            data.num_hyperedges = torch.tensor(
                [data.edge_index[0].max()-data.n_x[0]+1])
    
    if args.method in ['AllSetTransformer', 'AllDeepSets']:
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
        if args.exclude_self:
            data = expand_edge_index(data)
    
        #     Compute deg normalization: option in ['all_one','deg_half_sym'] (use args.normtype)
        # data.norm = torch.ones_like(data.edge_index[0])
        data = norm_contruction(data, option=args.normtype)
    elif args.method in ['CEGCN', 'CEGAT']:
        data = ExtractV2E(data)
        data = ConstructV2V(data)
        data = norm_contruction(data, TYPE='V2V')
    
    elif args.method in ['HyperGCN']:
        data = ExtractV2E(data)

    elif args.method in ['SheafHyperGCNDiag','SheafHyperGCNOrtho','SheafHyperGCNGeneral', 'SheafHyperGCNLowRank']:
        data = ExtractV2E(data)
    #    Make the first he_id to be 0
        data.edge_index[1] -= data.edge_index[1].min()

    elif args.method in ['HNHN']:
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
        H = ConstructH_HNHN(data)
        data = generate_norm_HNHN(H, data, args)
        data.edge_index[1] -= data.edge_index[1].min()
    
    elif args.method in ['HCHA', 'HGNN', 'DiagSheafs','OrthoSheafs', 'GeneralSheafs', 'LowRankSheafs', 
                            'EquivSetGNN', 'SheafEquivSetGNN_Diag', 'SheafEquivSetGNN_Ortho', 'SheafEquivSetGNN_General', 'SheafEquivSetGNN_LowRank',
                            'DiagSheafsDiffusion', 'OrthoSheafsDiffusion', 'GeneralSheafsDiffusion', 'LowRankSheafsDiffusion']:
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
    #    Make the first he_id to be 0
        data.edge_index[1] -= data.edge_index[1].min()

    
    #     Get splits
    split_idx_lst = []
    for run in range(args.runs):
        split_idx = rand_train_test_idx(
            data.y, train_prop=args.train_prop, valid_prop=args.valid_prop)
        split_idx_lst.append(split_idx)
    
    
    # # Part 2: Load model
    
    
    model = parse_method(args, data)
    # put things to device
    if args.cuda in [0, 1]:
        device = torch.device('cuda:'+str(args.cuda)
                              if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    model, data = model.to(device), data.to(device)
    
    num_params = count_parameters(model)
    print(f"Number of parameters: {num_params}")
    if args.wandb:
        wandb.watch(model)
    
    # # Part 3: Main. Training + Evaluation
    
    
    logger = Logger(args.runs, args)
    
    criterion = nn.NLLLoss()
    eval_func = eval_acc
    
    model.train()
    # print('MODEL:', model)
    
    ### Training loop ###
    runtime_list = []

    train_accs_runs = []
    valid_accs_runs = []
    test_accs_runs = []
    train_loss_runs = []
    valid_loss_runs = []
    test_loss_runs = []

    for run in tqdm(range(args.runs)):
        start_time = time.time()
        split_idx = split_idx_lst[run]
        train_idx = split_idx['train'].to(device)
     
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        best_val = float('-inf')
        train_accs_one_run = []
        valid_accs_one_run = []
        test_accs_one_run = []
        train_loss_one_run = []
        valid_loss_one_run = []
        test_loss_one_run = []
        for epoch in range(args.epochs):  
            # Training part
            model.train()
            optimizer.zero_grad()
            out = model(data)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out[train_idx], data.y[train_idx])
            loss.backward()
            optimizer.step()
    #         if args.method == 'HNHN':
    #             scheduler.step()
    #         Evaluation part
            result = evaluate(model, data, split_idx, eval_func)
            logger.add_result(run, result[:3])
    
            if epoch % args.display_step == 0 and args.display_step > 0:
                print(f'Epoch: {epoch:02d}, '
                      f'Train Loss: {loss:.4f}, '
                      f'Valid Loss: {result[4]:.4f}, '
                      f'Test  Loss: {result[5]:.4f}, '
                      f'Train Acc: {100 * result[0]:.2f}%, '
                      f'Valid Acc: {100 * result[1]:.2f}%, '
                      f'Test  Acc: {100 * result[2]:.2f}%')
            if args.wandb:
                train_accs_one_run.append(100 * result[0])
                valid_accs_one_run.append(100 * result[1])
                test_accs_one_run.append(100 * result[2])
                train_loss_one_run.append(loss.detach().cpu())
                valid_loss_one_run.append(result[4].detach().cpu())
                test_loss_one_run.append(result[5].detach().cpu())

            
            # if we want stats per run:
            # if args.wandb:
            #     log_corpus = {
            #         f'train_loss_run_{run}': loss,
            #         f'val_loss_run_{run}': result[4],
            #         f'test_loss_run_{run}': result[5],
            #         f'train_acc_run_{run}': 100 * result[0],
            #         f'test_acc_run_{run}': 100 * result[1],
            #         f'val_acc_run_{run}': 100 * result[2]
            #     }
            #     wandb.log(log_corpus)

        end_time = time.time()
        runtime_list.append(end_time - start_time)

        if args.wandb:
            #add training statistics from the crt running
            train_accs_runs.append(train_accs_one_run)
            valid_accs_runs.append(valid_accs_one_run)
            test_accs_runs.append(test_accs_one_run)
            train_loss_runs.append(train_loss_one_run)
            valid_loss_runs.append(valid_loss_one_run)
            test_loss_runs.append(test_loss_one_run)

        # logger.print_statistics(run)
    

    
    ### Save results ###
    avg_time, std_time = np.mean(runtime_list), np.std(runtime_list)

    if args.wandb:
        train_accs_runs = np.array(train_accs_runs)
        valid_accs_runs = np.array(valid_accs_runs)
        test_accs_runs = np.array(test_accs_runs)
        train_loss_runs = np.array(train_loss_runs)
        valid_loss_runs = np.array(valid_loss_runs)
        test_loss_runs = np.array(test_loss_runs)

        train_accs_mean = np.mean(train_accs_runs, 0)
        train_accs_std = np.std(train_accs_runs, 0)
        valid_accs_mean = np.mean(valid_accs_runs, 0)
        valid_accs_std = np.std(valid_accs_runs, 0)
        test_accs_mean = np.mean(test_accs_runs, 0)
        test_accs_std = np.std(test_accs_runs, 0)
        train_loss_mean = np.mean(train_loss_runs, 0)
        train_loss_std = np.std(train_loss_runs, 0)
        valid_loss_mean = np.mean(valid_loss_runs, 0)
        valid_loss_std = np.std(valid_loss_runs, 0)
        test_loss_mean = np.mean(test_loss_runs, 0)
        test_loss_std = np.std(test_loss_runs, 0)
        



        # pdb.set_trace()
        best_accuracy = -100
        for epoch in range(len(train_accs_mean)):
            best_accuracy = max(best_accuracy, valid_accs_mean[epoch])
            log_corpus = {
                f'train_accs_mean': train_accs_mean[epoch],
                f'val_accs_mean': valid_accs_mean[epoch],
                f'test_accs_mean': test_accs_mean[epoch],
                f'best_accs_mean': best_accuracy,

                f'train_acc_std': train_accs_std[epoch],
                f'test_acc_std': valid_accs_std[epoch],
                f'val_acc_std': test_accs_std[epoch],

                f'train_loss_mean': train_loss_mean[epoch],
                f'val_loss_mean': valid_loss_mean[epoch],
                f'test_loss_mean': test_loss_mean[epoch],

                f'train_loss_std': train_loss_std[epoch],
                f'test_loss_std': valid_loss_std[epoch],
                f'val_loss_std': test_loss_std[epoch],
            }
            wandb.log(log_corpus, step=epoch)

    best_val, best_test = logger.print_statistics()
    res_root = 'hyperparameter_tuning'
    if not osp.isdir(res_root):
        os.makedirs(res_root)

    filename = f'{res_root}/{args.dname}_.csv'
    print(f"Saving results to {filename}")
    with open(filename, 'a+') as write_obj:
        cur_line = f'{args.method}_{args.lr}_{args.wd}_{args.heads}'
        cur_line += f',{best_val.mean():.3f} ± {best_val.std():.3f}'
        cur_line += f',{best_test.mean():.3f} ± {best_test.std():.3f}'
        cur_line += f',{num_params}, {avg_time:.2f}s, {std_time:.2f}s' 
        cur_line += f',{avg_time//60}min{(avg_time % 60):.2f}s'
        cur_line += f'\n'
        write_obj.write(cur_line)

    all_args_file = f'{res_root}/all_args_{args.dname}_.csv'
    with open(all_args_file, 'a+') as f:
        f.write(str(args))
        f.write('\n')

    end_time = time.time()
    total_time = end_time - start_time
    print(f"TIME FOR ONE EXPERIMENT WITH {args.runs} RUNS: \\ Minutes: {total_time//60}, seconds {total_time%60}")
    print('All done! Exit python code')
    quit()
    

