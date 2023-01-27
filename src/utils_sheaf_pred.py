import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import pdb
from torch_scatter import scatter, scatter_mean, scatter_add
#these are some utilitary function. I will move them from here soon
def predict_blocks(x, e, hyperedge_index, sheaf_lin, args):
    # MLP(x_i, e_j) with e_j establish at initialisation
    # select all pairs (node, hyperedge)
    row, col = hyperedge_index
    xs = torch.index_select(x, dim=0, index=row)
    es = torch.index_select(e, dim=0, index=col)

    h_sheaf = torch.cat((xs,es), dim=-1) #sparse version of an NxEx2f tensor
    
    
    h_sheaf = sheaf_lin(h_sheaf)  #sparse version of an NxExd tensor
    if args.sheaf_act == 'sigmoid':
        h_sheaf = F.sigmoid(h_sheaf) # output d numbers for every entry in the incidence matrix
    elif args.sheaf_act == 'tanh':
        h_sheaf = F.tanh(h_sheaf) # output d numbers for every entry in the incidence matrix
    return h_sheaf

def predict_blocks_var2(x, hyperedge_index, sheaf_lin, args):
    #here hypergraph_att  is always the average of the intermediate x features
    row, col = hyperedge_index
    e = scatter_mean(x[row],col, dim=0)
    
    xs = torch.index_select(x, dim=0, index=row)
    es= torch.index_select(e, dim=0, index=col)


    # select all pairs (node, hyperedge)
    h_sheaf = torch.cat((xs,es), dim=-1) #sparse version of an NxEx2f tensor

    h_sheaf = sheaf_lin(h_sheaf)  #sparse version of an NxExd tensor


    if args.sheaf_act == 'sigmoid':
        h_sheaf = F.sigmoid(h_sheaf) # output d numbers for every entry in the incidence matrix
    elif args.sheaf_act == 'tanh':
        h_sheaf = F.tanh(h_sheaf) # output d numbers for every entry in the incidence matrix
    
    return h_sheaf

def predict_blocks_var3(x, hyperedge_index, sheaf_lin, args):
    #here hypergraph_att  is always the average of the intermediate x features
    row, col = hyperedge_index
    e = scatter_mean(x[row],col, dim=0)
    
    xs = torch.index_select(x, dim=0, index=row)
    es= torch.index_select(e, dim=0, index=col)

    # select all pairs (node, hyperedge)
    h_sheaf = torch.cat((xs,es), dim=-1) #sparse version of an NxEx2f tensor

    
    h_sheaf = sheaf_lin(h_sheaf)  #sparse version of an NxExd tensor
    if args.sheaf_act== 'sigmoid':
        h_sheaf = F.sigmoid(h_sheaf) # output d numbers for every entry in the incidence matrix
    elif args.sheaf_act == 'tanh':
        h_sheaf = F.tanh(h_sheaf) # output d numbers for every entry in the incidence matrix
    
    return h_sheaf

def predict_blocks_transformer(x, hyperedge_index, transformer_layer, transformer_lin_layer, args):
    row, col = hyperedge_index

    receivers_idx = list(range(len(row))) # 1 2 3.. nnz
    # receivers_val = x[row] # x[0] x[1] x[0] x[1] x[3] .. 
    receivers_nodes = row.detach().cpu().numpy()
    receivers_hedge = col.detach().cpu().numpy() # 0 0 1 1 1 2 2


    receivers_pairs = list(zip(receivers_idx, receivers_nodes, receivers_hedge))
    key_func = lambda x: x[2]
    receivers_pairs_sort = sorted(receivers_pairs, key=key_func) #rearrange the tuples to be in asc order by receivers_group

    # x_row_sorted = np.array(receivers_pairs_sort)[:,1]
    # x_col_sorted = np.array(receivers_pairs_sort)[:,2]

    edges = []
    for key, group in itertools.groupby(receivers_pairs_sort, key_func):
        aa = np.array(list(group))
        aa = aa[:,0]
        #for each group create all-to-all combinations of edges (fully connected per hyperdge)
        # edges = edges + (list(itertools.product(aa, repeat=2)))
        edges = edges + (list(itertools.permutations(aa, 2)))

    device = torch.device('cuda:'+str(args.cuda)
                              if torch.cuda.is_available() else 'cpu')
    edges = torch.tensor(np.array(edges)).t().to(device)
    transformer_input = x[row] #nnz x f in the original order

    # print(transformer_input.mean())
    transformer_output = transformer_layer(transformer_input, edge_index=edges)
    # print(transformer_output.mean())

    h_sheaf = torch.cat((transformer_input, transformer_output), -1)

    h_sheaf = transformer_lin_layer(h_sheaf)
    if args.sheaf_act == 'sigmoid':
        h_sheaf = F.sigmoid(h_sheaf) # output d numbers for every entry in the incidence matrix
    elif args.sheaf_act == 'tanh':
        h_sheaf = F.tanh(h_sheaf) # output d numbers for every entry in the incidence matrix
    return h_sheaf