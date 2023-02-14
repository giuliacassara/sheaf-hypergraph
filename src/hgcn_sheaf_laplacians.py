from utils import *
import torch, math, numpy as np, scipy.sparse as sp
import torch.nn as nn, torch.nn.functional as F, torch.nn.init as init
import ipdb

from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

import numpy as np
import itertools
import time

from torch_scatter import scatter, scatter_mean, scatter_add


def reduce_graph(X_reduced, m, d, edge_index):
    """
    X_reduced: nnz x d x f

    given some features X extract for each hyperedge in edge_index
    just edges connecting min to max

    If m also connect them to mediators.

    Return:
    edges_idx: idx in nnz array for edges for the graph assoc with the hyperedge: used to select F_v<e F_w<e
    edges_idx_diag: idx in the nnz array for edges corresp to the diagonal used to select F_v<e F_v<e^T
    all_contained_hyperedges: idx for the hyperedges containing each node (help in the scatter phase): used to compute \sum_e F_v<e F_v<e^T
    hgcn_edges: all the new hyperedges
    """
    # X_reduces: nx x f
    edges, weights = [], {}

    #TODO: is this what we really want?? a random projection each time?
    rv = torch.rand(X_reduced.shape[-1]).to(X_reduced.device).unsqueeze(-1)
    # print("rv", rv.mean())

    row, col = edge_index
    
    receivers_idx = torch.range(0, len(row)-1).to(X_reduced.device) #to keep track of position in the nnz vector
    receivers_nodes = row # nodes that are part of the hyperedge
    receivers_hedge = col # index of the hyperedge 0 0 0 1 1 2 2 2

    receivers_pairs = torch.stack((receivers_idx, receivers_nodes, receivers_hedge),dim=-1)
    key_func = lambda x: x[2] #sort them according to the hyperedge such that we can extract the nodes from each hyperedge

    receivers_pairs = receivers_pairs.detach().cpu().numpy()
    receivers_pairs_sort = sorted(receivers_pairs, key=key_func) #rearrange the tuples to be in asc order by receivers_group
    
    # COMPUTE argmax(F_v<e(X_v)) and argmin(F_v<e(X_v)) SUCH THAT WE CAN SELECT THE EDGE
    X_hedge = X_reduced #n x d x f
    X_hedge = X_hedge.reshape(X_hedge.shape[0]*d, X_hedge.shape[-1]) #nd x f

    p = X_hedge @ rv    #dot product on second axis to get rid of the channels --  preserve the idea from HyperGCN
    p = p.squeeze(-1)

    p = p.reshape(row.shape[0], d) # nnz x d
    p = torch.transpose(p,0,1) # d x nnz
    p_1 = p.unsqueeze(-1) # d x nnz x 1
    p_2 = p.unsqueeze(-2) # d x 1 x nnz
    p_dist = p_1 - p_2 # d x nnz x nnz
    p_dist = torch.permute(p_dist,(1,2,0)) # nnz x nnz x d
    p_dist = torch.linalg.norm(p_dist,dim=-1, ord=2)  # nnz x nnz
    p_dist = p_dist.detach().cpu().numpy()


    edges = []
    edges_idx = []
    edges_idx_m = []

    for key, group in itertools.groupby(receivers_pairs_sort, key_func):
        hyperedge = np.array(list(group)).astype(int)
        hyperedge_nodes =  hyperedge[:,1] # nodes that are part of the crt hyperedge
        hyperedge_pos = hyperedge[:,0] # position in the nnz vector corresponding to each (node, hyperedge) pair for the crt hyperedge

        # extract the distance submatrix corresponding to the crt hyperedge
        p_dist_partial = np.take(p_dist, axis=0, indices=hyperedge_pos) 
        p_dist_partial = np.take(p_dist_partial, axis=1, indices=hyperedge_pos)

        # find the 2 nodes with maximum distance 
        # different from HyperGCN the distance computed in the input space X_v
        # but on the opinion/sheaf space (F_v<e(X_v))
        s, i = np.unravel_index(np.argmax(p_dist_partial), p_dist_partial.shape)

        Se, Ie = hyperedge_nodes[s], hyperedge_nodes[i] #these are the nodes that are part of the hyperedge. useful when drawing the edge
        S_idx, I_idx = hyperedge_pos[s], hyperedge_pos[i] #these are the position in the  nnz vector: (node, edge) pair. Usefull when extracting sheaf features corresponding to  pairs

        # two stars with mediators
        c = 2*len(hyperedge_pos) - 3    # normalisation constant
 
        if m:
            # connect the supremum (Se) with the infimum (Ie)
            edges.extend([[Se, Ie], [Ie, Se]])
            #also keep track of indexes in the nnz vector such that we can extract the sheaf F
            edges_idx.extend([[S_idx, I_idx], [I_idx, S_idx]])

            if (Se,Ie) not in weights:
                weights[(Se,Ie)] = 0
            weights[(Se,Ie)] += float(1/c)

            if (Ie,Se) not in weights:
                weights[(Ie,Se)] = 0
            weights[(Ie,Se)] += float(1/c)

            # # connect the supremum (Se) and the infimum (Ie) with each mediator
            for mediator_idx, mediator_e in zip(hyperedge_pos, hyperedge_nodes):
                if mediator_e != Se and mediator_e != Ie:
                    edges.extend([[Se,mediator_e], [mediator_e, Se], [Ie, mediator_e], [mediator_e,Ie]])
                    #also keep track of indexes in the nnz vector such that we can extract the sheaf F
                    edges_idx.extend([[S_idx, mediator_idx], [mediator_idx, S_idx], [I_idx, mediator_idx], [mediator_idx, I_idx]])
                    weights = update(Se, Ie, mediator_e, weights, c)
                    
        else:
            edges.extend([[Se,Ie], [Ie,Se]])
            #also keep track of indexes in the nnz vector such that we can extract the sheaf F
            edges_idx.extend([[S_idx, I_idx], [I_idx, S_idx]])

            e = len(hyperedge_pos)
            
            if (Se,Ie) not in weights:
                weights[(Se,Ie)] = 0
            weights[(Se,Ie)] += float(1/e)

            if (Ie,Se) not in weights:
                weights[(Ie,Se)] = 0
            weights[(Ie,Se)] += float(1/e)    

    # ADDING THE SELF LOOP
    # for each node v we will add on the diagonal \sum_e F_v<e(X_v)^T F_v<e(X_v)
    receivers_pairs_diag = torch.stack((receivers_idx, receivers_nodes, receivers_hedge),dim=-1).detach().cpu().numpy()
    key_func_diag = lambda x: x[1] #this type sort according to the nodes
    receivers_pairs_sort_diag = sorted(receivers_pairs_diag, key=key_func_diag) #rearrange the tuples to be in asc order by receivers_group

    edges_idx_diag = []
    all_contained_hyperedges = []
    

    idx = 0
    for key, group in itertools.groupby(receivers_pairs_sort_diag, key_func_diag):
        # see all hyperedges e one node is part of:
        contained_hyperedges = np.array(list(group)).astype(int)
        node_idx = contained_hyperedges[:,1][0] #all node idxes should be the same due to grouping
        contained_hyperedges = contained_hyperedges[:,0] #indexes in the nnz vectors for all (node, e) when node is part of hyperedge e
        
        edges.extend([[node_idx, node_idx]])
        edges_idx_diag.extend(list(zip(contained_hyperedges,contained_hyperedges)))
        all_contained_hyperedges.extend([idx]*len(contained_hyperedges)) #keep track of what elements I need to aggergate for each node
        idx = idx+1

    edges_idx = torch.tensor(np.array(edges_idx).transpose()).to(X_reduced.device)
    edges_idx_diag = torch.tensor(np.array(edges_idx_diag).transpose()).to(X_reduced.device)

    all_contained_hyperedges = torch.tensor(np.array(all_contained_hyperedges).astype(int)).to(X_reduced.device)
    hgcn_edges = torch.tensor(np.array(edges).transpose()).to(X_reduced.device)

    return edges_idx, edges_idx_diag, all_contained_hyperedges, hgcn_edges

    

def SheafLaplacianDiag(H, m, d, edge_index, sheaf, E=None):
    """
    approximates the E defined by the E Laplacian with/without mediators

    arguments:
    H: (num_nodes*d) x f
    m: true or False for using mediators
    d: dim of stack
    edge_index: 2xnnz
    sheaf: nnz x d (for diagonal sheaf)
    """
    
    F = sheaf
    num_nodes = H.shape[0]//d
    MLP_hidden = H.shape[-1]

    #APPLY THE REDUCTION SHEAF() to features X -> F_v<e(X_v) for each pair (v,e)

    H_selected =  H.view((num_nodes,d, -1)) # n x d x f
    H_selected = torch.index_select(H_selected, dim=0, index=edge_index[0]) #select nodes corresponding to the pairs in nnz
    X_reduced = H_selected.permute(0,2,1) # nnz x f x d
    sheaf = torch.diag_embed(sheaf, dim1=-2, dim2=-1) #nnz x d x d
  
    sheaf = sheaf.unsqueeze(1).repeat(1,MLP_hidden,1,1).reshape(-1, d, d) # nnz f x d x d
    
    X_reduced  = X_reduced.reshape(-1, d).unsqueeze(1) #nnz f x d
    # TODO think if this is ok??
    X_reduced = torch.bmm(X_reduced, sheaf) # nnz f x d
    X_reduced = X_reduced.reshape(-1, MLP_hidden , d) #nnz x f x d
    X_reduced = X_reduced.permute(0,2,1)  # nnz x d x f

    #return edges_idx: idx in nnz array for edges for the graph assoc with the hyperedge 
    # edges_idx_diag: idx in the nnz array for edges corresp to the diagonal
    # all_contained_hyperedges: idx for the hyperedges containing each node (help in the scatter phase)
    # hgcn_edges: all the new hyperedges
    edges_idx, edges_idx_diag, all_contained_hyperedges, hgcn_edges = reduce_graph(X_reduced, m, d, edge_index) 

    # compute -F_v<e(X_v)^T F_w<e(X_w) for all (v, w) non-diagonal pairs
    F_source = torch.index_select(F, dim=0, index=edges_idx[0]) # F_v<e(X_v)
    F_dest = torch.index_select(F, dim=0, index=edges_idx[1]) # F_w<e(X_v)
    attributes = -F_source * F_dest # -F_v<e(X_v)^T F_w<e(X_w) (for diagonal elementwise is enough)

    # compute \sum_e F_v<e(X_v)^T F_v<e(X_v) for all edges e containing v      
    F_source_diag = torch.index_select(F, dim=0, index=edges_idx_diag[0]) 
    F_dest_diag = torch.index_select(F, dim=0, index=edges_idx_diag[1])
    attributes_diag = F_source_diag * F_dest_diag
    #for each selfloop (x,x) aggregate the reduction for all hyperedges that node is part of sum_e F_v<e(X_v)^T F_v<e(X_v)
    attributes_diag = scatter_add(attributes_diag, all_contained_hyperedges, dim=0) 
    attributes = torch.concat([attributes, attributes_diag], axis=0) #combine non-diagonal and diagonal elements of the laplacian

    # compute the indezes for the large block sheaf laplacian nxn -> nd x nd
    d_range = torch.arange(d, device=H.device).view(1,-1,1).repeat(2,1,1) #2xdx1
    hgcn_edges = hgcn_edges.unsqueeze(1) #2x1xK
    hgcn_edges = d * hgcn_edges + d_range #2xdxK
    hgcn_edges = hgcn_edges.permute(0,2,1).reshape(2,-1) #2x(d*K)

    h_sheaf_index = hgcn_edges
    h_sheaf_attributes = attributes.view(-1)

    #be careful these contains duplicate attributes with different weights. 
    # in the end the sparse matrix will sum them anyway
    return h_sheaf_index, h_sheaf_attributes

def SheafLaplacianGeneral(H, m, d, edge_index, sheaf, E=None):
    """
    approximates the E defined by the E Laplacian with/without mediators

    arguments:
    H: (num_nodes*d) x f
    m: True or False use or not mediatos 
    edge_index: 2xnnz 
    sheaf: nnz x d x d (for general sheaf)
    """
    
    time0=time.time()
    F = sheaf                   #nnz x (d*d) (for general sheaf)
    num_nodes = H.shape[0]//d
    MLP_hidden = H.shape[-1]

    #APPLY THE REDUCTION SHEAF() to features X -> F_v<e(X_v) for each pair (v,e)

    H_selected =  H.view((num_nodes,d, -1)) # n x d x f
    H_selected = torch.index_select(H_selected, dim=0, index=edge_index[0]) #select nodes corresponding to the pairs in nnz
    X_reduced = H_selected.view((H_selected.shape[0], d, -1)) #nnz x d x f
    X_reduced = X_reduced.permute(0,2,1) #nnz x f x d
    # X_reduced = X_reduced.view((H_selected.shape[0],-1, d)) # n x f x d

    sheaf = sheaf.view(sheaf.shape[0],d,d) #nnz x d x d
    sheaf = sheaf.unsqueeze(1).repeat(1,MLP_hidden,1,1).view(-1, d, d) # nnz f x d x d
    X_reduced  = X_reduced.reshape(-1, d).unsqueeze(1) #nnz f x 1 x d 
    X_reduced = torch.bmm(X_reduced, sheaf) #nnz fx 1 x d
    X_reduced = X_reduced.reshape(-1, MLP_hidden , d) #nnz x f x d
    X_reduced = X_reduced.permute(0,2,1)  # nnz x d x f

    #return edges_idx: idx in nnz array for edges for the graph assoc with the hyperedge 
    # edges_idx_diag: idx in the nnz array for edges corresp to the diagonal
    # all_contained_hyperedges: idx for the hyperedges containing each node (help in the scatter phase)
    # hgcn_edges: all the new hyperedges
    edges_idx, edges_idx_diag, all_contained_hyperedges, hgcn_edges = reduce_graph(X_reduced, m, d, edge_index)

    # compute -F_v<e(X_v)^T F_w<e(X_w) for all (v, w) non-diagonal pairs
    F = F.view(F.shape[0],d,d) # nnz x d x d, all the linear projections for all (node, edge) pairs
    F_source = torch.index_select(F, dim=0, index=edges_idx[0]) # F_v<e(X_v)
    F_dest = torch.index_select(F, dim=0, index=edges_idx[1]) # F_v<e(X_v)
    attributes = -1 * torch.bmm(F_source.transpose(1,2),F_dest) # -F_v<e(X_v)^T F_w<e(X_w)

    # compute \sum_e F_v<e(X_v)^T F_v<e(X_v) for all edges e containing v  
    F_source_diag = torch.index_select(F, dim=0, index=edges_idx_diag[0]) 
    F_dest_diag = torch.index_select(F, dim=0, index=edges_idx_diag[1])
    attributes_diag = torch.bmm(F_source_diag.transpose(1,2),F_dest_diag)

    #for each selfloop (x,x) aggregate the reduction for all hyperedges that node is part of sum_e F_v<e(X_v)^T F_v<e(X_v)
    attributes_diag = scatter_add(attributes_diag, all_contained_hyperedges, dim=0)
    attributes = torch.concat([attributes, attributes_diag], axis=0)

    # TODO: at some point take this out as a function
    # compute the indexes for the large block sheaf laplacian nxn -> nd x nd
    d_range = torch.arange(d, device=H.device)
    d_range_edges = d_range.repeat(d).view(-1,1) #0,1..d,0,1..d..   d*d elems
    d_range_nodes = d_range.repeat_interleave(d).view(-1,1) #0,0..0,1,1..1..d,d..d  d*d elems
    hgcn_edges = hgcn_edges.unsqueeze(1) 

    hyperedge_index_0 = d * hgcn_edges[0] + d_range_nodes
    hyperedge_index_0 = hyperedge_index_0.permute((1,0)).reshape(1,-1)
    hyperedge_index_1 = d * hgcn_edges[1] + d_range_edges
    hyperedge_index_1 = hyperedge_index_1.permute((1,0)).reshape(1,-1)
    hgcn_edges = torch.concat((hyperedge_index_0, hyperedge_index_1), 0)

    h_sheaf_index = hgcn_edges
    h_sheaf_attributes = attributes.view(-1)
    #be careful these contains duplicate attributes with different weights. 
    # in the end the sparse matrix will sum them anyway (collate())
    return h_sheaf_index, h_sheaf_attributes

def SheafLaplacianOrtho(H, m, d, edge_index, sheaf, E=None):
    """
    approximates the E defined by the E Laplacian with/without mediators

    arguments:
    H: (num_nodes*d) x f
    m: True or False use or not mediatos 
    edge_index: 2xnnz 
    sheaf: nnz x d x d (for general sheaf)
    """
    
    F = sheaf                   #nnz x (d*d) (for ortho sheaf)
    num_nodes = H.shape[0]//d
    MLP_hidden = H.shape[-1]

    #APPLY THE REDUCTION SHEAF() to features X -> F_v<e(X_v) for each pair (v,e)

    H_selected =  H.view((num_nodes,d, -1)) # n x d x f
    H_selected = torch.index_select(H_selected, dim=0, index=edge_index[0]) # nnz x d x f
    X_reduced = H_selected.view((H_selected.shape[0], d, -1)) # nnz x d x f
    X_reduced = X_reduced.permute(0,2,1) # nnz x f x d
    # X_reduced = X_reduced.view((H_selected.shape[0],-1, d)) # n x f x d

    sheaf = sheaf.view(sheaf.shape[0],d,d) #nnz x d x d containinf all the projections for F_v<e
    sheaf = sheaf.unsqueeze(1).repeat(1,MLP_hidden,1,1).view(-1, d, d) # nnz x f x d x d
    X_reduced  = X_reduced.reshape(-1, d).unsqueeze(1) #nnz f x 1 x d
    X_reduced = torch.bmm(X_reduced, sheaf) #nnz f x 1 x d
    X_reduced = X_reduced.reshape(-1, MLP_hidden , d) #nnz x f x d
    X_reduced = X_reduced.permute(0,2,1) #nnz x d x f

    #return edges_idx: idx in nnz array for edges for the graph assoc with the hyperedge 
    # edges_idx_diag: idx in the nnz array for edges corresp to the diagonal
    # all_contained_hyperedges: idx for the hyperedges containing each node (help in the scatter phase)
    # hgcn_edges: all the new hyperedges
    edges_idx, edges_idx_diag, all_contained_hyperedges, hgcn_edges = reduce_graph(X_reduced, m, d, edge_index)

    # compute -F_v<e(X_v)^T F_w<e(X_w) for all (v, w) non-diagonal pairs
    F = F.view(F.shape[0],d,d) # nnz x d x d all the linear projections for all (node, edge) pairs
    F_source = torch.index_select(F, dim=0, index=edges_idx[0]) # F_v<e(X_v)
    F_dest = torch.index_select(F, dim=0, index=edges_idx[1]) # F_v<e(X_v)
    attributes = -1 * torch.bmm(F_source.transpose(1,2),F_dest) # -F_v<e(X_v)^T F_w<e(X_w)

    # normally we compute \sum_e F_v<e(X_v)^T F_v<e(X_v) for all edges e containing v  
    #NOTE: For ORTHOGONAL matrices we don't need to compute F_s^T @ F_D since it is always I
    attributes_diag = torch.eye(d).unsqueeze(0).repeat((edges_idx_diag.shape[1],1,1)).to(H.device) # nnz x d x d
    #for each selfloop (x,x) aggregate the reduction for all hyperedges that node is part of sum_e F_v<e(X_v)^T F_v<e(X_v)
    attributes_diag = scatter_add(attributes_diag, all_contained_hyperedges, dim=0) #
    attributes = torch.concat([attributes, attributes_diag], axis=0)

    # TODO: at some point take this out as a function
    # compute the indexes for the large block sheaf laplacian nxn -> nd x nd

    d_range = torch.arange(d, device=H.device)
    d_range_edges = d_range.repeat(d).view(-1,1) #0,1..d,0,1..d..   d*d elems
    d_range_nodes = d_range.repeat_interleave(d).view(-1,1) #0,0..0,1,1..1..d,d..d  d*d elems
    hgcn_edges = hgcn_edges.unsqueeze(1) 

    hyperedge_index_0 = d * hgcn_edges[0] + d_range_nodes
    hyperedge_index_0 = hyperedge_index_0.permute((1,0)).reshape(1,-1)
    hyperedge_index_1 = d * hgcn_edges[1] + d_range_edges
    hyperedge_index_1 = hyperedge_index_1.permute((1,0)).reshape(1,-1)
    hgcn_edges = torch.concat((hyperedge_index_0, hyperedge_index_1), 0)

    h_sheaf_index = hgcn_edges
    h_sheaf_attributes = attributes.view(-1)

    #be careful these contains duplicate attributes with different weights. 
    # in the end the sparse matrix will sum them anyway
    return h_sheaf_index, h_sheaf_attributes