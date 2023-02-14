from models import MLP
import torch.nn as nn
import torch
import torch_geometric
from orthogonal import Orthogonal
from torch_scatter import scatter, scatter_mean, scatter_add
import utils

import torch.nn.functional as F
import itertools
import numpy as np
import pdb

class SheafBuilderDiag(nn.Module):
    def __init__(self, args):
        super(SheafBuilderDiag, self).__init__()
        self.args = args
        self.prediction_type = args.sheaf_pred_block
        self.sheaf_dropout = args.sheaf_dropout
        self.special_head = args.sheaf_special_head
        self.d = args.heads
        self.MLP_hidden = args.MLP_hidden
        self.norm = args.AllSet_input_norm

        if self.prediction_type == 'transformer':
            transformer_head = args.sheaf_transformer_head
            self.transformer_layer=torch_geometric.nn.TransformerConv(
                                in_channels=self.MLP_hidden, 
                                out_channels=self.MLP_hidden//transformer_head,
                                heads=transformer_head)
                                
            self.transformer_lin_layer = MLP(
                        in_channels=self.MLP_hidden + self.MLP_hidden, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=self.d,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
                        
        else:
            self.sheaf_lin = MLP(
                        in_channels=2*self.MLP_hidden, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=self.d,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
        if self.prediction_type == 'MLP_var3':
            self.sheaf_lin2 = MLP(
                        in_channels=self.MLP_hidden, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=args.MLP_hidden,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
                        

    def reset_parameters(self):
        if self.prediction_type == 'transformer':
            self.transformer_lin_layer.reset_parameters()
            self.transformer_layer.reset_parameters()
        elif self.prediction_type == 'MLP_var3':
            self.sheaf_lin.reset_parameters()
            self.sheaf_lin2.reset_parameters()
        else:
            self.sheaf_lin.reset_parameters()
        

    #this is exclusively for diagonal sheaf
    def forward(self, x, e, hyperedge_index):
        """ tmp
        x: Nd x f -> N x f
        e: Ed x f -> E x f
        -> (concat) N x E x (d+1)F -> (linear project) N x E x d (the elements on the diagonal of each dxd block)
        -> (reshape) (Nd x Ed) with NxE diagonal blocks of dimension dxd

        """

        pdb.set_trace()
        num_nodes = hyperedge_index[0].max().item() + 1
        num_edges = hyperedge_index[1].max().item() + 1
        x = x.view(num_nodes, self.d, x.shape[-1]).mean(1) # N x d x f -> N x f
        e = e.view(num_edges, self.d, e.shape[-1]).mean(1) # # x d x f -> E x f

        # h_sheaf = self.predict_blocks(x, e, hyperedge_index, sheaf_lin)
        # h_sheaf = self.predict_blocks_var2(x, hyperedge_index, sheaf_lin)
        if self.prediction_type == 'transformer':
            h_sheaf = predict_blocks_transformer(x, hyperedge_index, self.transformer_layer, self.transformer_lin_layer, self.args)
        elif self.prediction_type == 'MLP_var1':
            h_sheaf = predict_blocks(x, e, hyperedge_index, self.sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var2':
            h_sheaf = predict_blocks_var2(x, hyperedge_index, self.sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var3':
            h_sheaf = predict_blocks_var3(x, hyperedge_index, self.sheaf_lin,  self.sheaf_lin2, self.args)

        if self.sheaf_dropout:
            h_sheaf = F.dropout(h_sheaf, p=self.dropout, training=self.training)
        #add a head having just 1 on the diagonal. this should be similar to the normal hypergraph conv
        if self.special_head:
            new_head_mask = [1]*(self.d-1) + [0]
            new_head = [0]*(self.d-1) + [1]
            h_sheaf = h_sheaf * torch.tensor(new_head_mask, device=x.device) + torch.tensor(new_head, device=x.device)
        
        self.h_sheaf = h_sheaf #this is stored in self for testing purpose

        # from a d-dim tensor assoc to every entrence in edge_index
        # create a sparse incidence Nd x Ed

        # We need to modify indices from the NxE matrix 
        # to correspond to the large Nd x Ed matrix, but restrict only on the element of the diagonal of each block
        # indices: scalar [i,j] -> block dxd with indices [d*i, d*i+1.. d*i+d-1; d*j, d*j+1 .. d*j+d-1]
        # attributes: reshape h_sheaf

        d_range = torch.arange(self.d, device=x.device).view(1,-1,1).repeat(2,1,1) #2xdx1
        hyperedge_index = hyperedge_index.unsqueeze(1) #2x1xK
        hyperedge_index = self.d * hyperedge_index + d_range #2xdxK

        hyperedge_index = hyperedge_index.permute(0,2,1).reshape(2,-1) #2x(d*K)

        h_sheaf_index = hyperedge_index
        h_sheaf_attributes = h_sheaf.reshape(-1) #(d*K)

        #the resulting (index, values) pair correspond to the diagonal of each block sub-matrix
        return h_sheaf_index, h_sheaf_attributes
        
    

class SheafBuilderGeneral(nn.Module):
    def __init__(self, args):
        super(SheafBuilderGeneral, self).__init__()
        self.args = args
        self.prediction_type = args.sheaf_pred_block
        self.sheaf_dropout = args.sheaf_dropout
        self.special_head = args.sheaf_special_head
        self.d = args.heads
        self.MLP_hidden = args.MLP_hidden
        self.norm = args.AllSet_input_norm
        self.norm_type = args.sheaf_normtype

        if self.prediction_type == 'transformer':
            transformer_head = args.sheaf_transformer_head
            self.transformer_layer = torch_geometric.nn.TransformerConv(
                        in_channels=self.MLP_hidden, 
                        out_channels=self.MLP_hidden//transformer_head,
                        heads=transformer_head)

            self.transformer_lin_layer = MLP(
                    in_channels=2*self.MLP_hidden, 
                    hidden_channels=args.MLP_hidden,
                    out_channels=self.d*self.d,
                    num_layers=1,
                    dropout=0.0,
                    Normalization='ln',
                    InputNorm=self.norm)
            
        else:
            self.general_sheaf_lin = MLP(
                    in_channels=2*self.MLP_hidden, 
                    hidden_channels=args.MLP_hidden,
                    out_channels=self.d*self.d,
                    num_layers=1,
                    dropout=0.0,
                    Normalization='ln',
                    InputNorm=self.norm)
        if self.prediction_type == 'MLP_var3':
            self.general_sheaf_lin2 = MLP(
                        in_channels=self.MLP_hidden, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=args.MLP_hidden,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)

    def reset_parameters(self):
        if self.prediction_type == 'transformer':
            self.transformer_lin_layer.reset_parameters()
            self.transformer_layer.reset_parameters()
        elif self.prediction_type == 'MLP_var3':
            self.general_sheaf_lin.reset_parameters()
            self.general_sheaf_lin2.reset_parameters()
        else:
            self.general_sheaf_lin.reset_parameters()

    def forward(self, x, e, hyperedge_index, debug=False):
        """ 
        x: N x f
        e: N x f 
        -> (concat) N x E x 2f -> (linear project) N x E x d*d
        -> (reshape) (Nd x Ed) with each block dxd being unconstrained

        """
        num_nodes = hyperedge_index[0].max().item() + 1
        num_edges = hyperedge_index[1].max().item() + 1
        x = x.view(num_nodes, self.d, x.shape[-1]).mean(1) # N x d x f -> N x f
        e = e.view(num_edges, self.d, e.shape[-1]).mean(1) # N x d x f -> N x f

        row, col = hyperedge_index
        x_row = torch.index_select(x, dim=0, index=row)
        e_col = torch.index_select(e, dim=0, index=col)

        if self.prediction_type == 'transformer':
            h_general_sheaf = predict_blocks_transformer(x, hyperedge_index, self.transformer_layer, self.transformer_lin_layer, self.args)
        elif self.prediction_type == 'MLP_var1':
            h_general_sheaf = predict_blocks(x, e, hyperedge_index, self.general_sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var2':
            h_general_sheaf = predict_blocks_var2(x, hyperedge_index, self.general_sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var3':
            h_general_sheaf = predict_blocks_var3(x, hyperedge_index, self.general_sheaf_lin, self.general_sheaf_lin2, self.args)
        #Iulia: Temporary debug
        # pdb.set_trace()
        # h_general_sheaf = h_general_sheaf.reshape(-1,self.d,self.d) * torch.eye(self.d, device=self.device)
        # h_general_sheaf = h_general_sheaf.reshape(-1,self.d*self.d)
        if debug:
            self.h_general_sheaf = h_general_sheaf #for debug purpose
        if self.sheaf_dropout:
            h_general_sheaf = F.dropout(h_general_sheaf, p=self.dropout, training=self.training)
        if (debug==True):
            print("x", x.mean(-1))
            print("x_row", x_row.mean(-1))
            print("e", e.mean(-1))
            print("e_col", e_col.mean(-1))
            print(hyperedge_index)

        # from a d-dim tensor assoc to every entrence in edge_index
        # create a sparse incidence Nd x Ed

        # modify indices to correspond to the big matrix and assign the weights
        # indices: [i,j] -> [d*i, d*i.. d*i+d-1, d*i+d-1; d*j, d*j+1 .. d*j, d*j+1,..d*j+d-1]
        
        d_range = torch.arange(self.d, device=x.device)
        d_range_edges = d_range.repeat(self.d).view(-1,1) #0,1..d,0,1..d..   d*d elems
        d_range_nodes = d_range.repeat_interleave(self.d).view(-1,1) #0,0..0,1,1..1..d,d..d  d*d elems
        hyperedge_index = hyperedge_index.unsqueeze(1) 
   

        hyperedge_index_0 = self.d * hyperedge_index[0] + d_range_nodes
        hyperedge_index_0 = hyperedge_index_0.permute((1,0)).reshape(1,-1)
        hyperedge_index_1 = self.d * hyperedge_index[1] + d_range_edges
        hyperedge_index_1 = hyperedge_index_1.permute((1,0)).reshape(1,-1)
        h_general_sheaf_index = torch.concat((hyperedge_index_0, hyperedge_index_1), 0)

        if self.norm_type == 'block_norm':
            # pass
            h_general_sheaf_1 = h_general_sheaf.reshape(h_general_sheaf.shape[0], self.d, self.d)
            num_nodes = hyperedge_index[0].max().item() + 1
            num_edges = hyperedge_index[1].max().item() + 1

            to_be_inv_nodes = torch.bmm(h_general_sheaf_1, h_general_sheaf_1.permute(0,2,1)) 
            to_be_inv_nodes = scatter_add(to_be_inv_nodes, row, dim=0, dim_size=num_nodes)

            to_be_inv_edges = torch.bmm(h_general_sheaf_1.permute(0,2,1), h_general_sheaf_1)
            to_be_inv_edges = scatter_add(to_be_inv_edges, col, dim=0, dim_size=num_edges)


            d_sqrt_inv_nodes = utils.batched_sym_matrix_pow(to_be_inv_nodes, -1.0) #n_nodes x d x d
            d_sqrt_inv_edges = utils.batched_sym_matrix_pow(to_be_inv_edges, -1.0) #n_edges x d x d
            

            d_sqrt_inv_nodes_large = torch.index_select(d_sqrt_inv_nodes, dim=0, index=row)
            d_sqrt_inv_edges_large = torch.index_select(d_sqrt_inv_edges, dim=0, index=col)


            alpha_norm = torch.bmm(d_sqrt_inv_nodes_large, h_general_sheaf_1)
            alpha_norm = torch.bmm(alpha_norm, d_sqrt_inv_edges_large)
            h_general_sheaf = alpha_norm.clamp(min=-1, max=1)
            h_general_sheaf = h_general_sheaf.reshape(h_general_sheaf.shape[0], self.d*self.d)

        #!!! Is this the correct reshape??? Please check!!
        h_general_sheaf_attributes = h_general_sheaf.reshape(-1)
        #create the big matrix from the dxd blocks  
        return h_general_sheaf_index, h_general_sheaf_attributes

class SheafBuilderOrtho(nn.Module):
    def __init__(self, args):
        super(SheafBuilderOrtho, self).__init__()
        self.args = args
        self.prediction_type = args.sheaf_pred_block
        self.sheaf_dropout = args.sheaf_dropout
        self.special_head = args.sheaf_special_head
        self.d = args.heads
        self.MLP_hidden = args.MLP_hidden
        self.norm = args.AllSet_input_norm

        self.orth_transform = Orthogonal(d=self.d, orthogonal_map='householder') #method applied to transform params into ortho dxd matrix

        transformer_head = args.sheaf_transformer_head
        if self.prediction_type == 'transformer':
            self.transformer_layer = torch_geometric.nn.TransformerConv(
                                        in_channels=self.MLP_hidden, 
                                        out_channels=self.MLP_hidden//transformer_head,
                                        heads=transformer_head)
            self.transformer_lin_layer = MLP(
                    in_channels=2*self.MLP_hidden, 
                    hidden_channels=args.MLP_hidden,
                    out_channels=self.d*(self.d-1)//2,
                    num_layers=1,
                    dropout=0.0,
                    Normalization='ln',
                    InputNorm=self.norm
                    )
        else:
            self.orth_sheaf_lin = MLP(
                    in_channels=2*self.MLP_hidden, 
                    hidden_channels=args.MLP_hidden,
                    out_channels=self.d*(self.d-1)//2,
                    num_layers=1,
                    dropout=0.0,
                    Normalization='ln',
                    InputNorm=self.norm)
        if self.prediction_type == 'MLP_var3':
            self.orth_sheaf_lin2 = MLP(
                        in_channels=self.MLP_hidden, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=args.MLP_hidden,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)

    def reset_parameters(self):
        if self.prediction_type == 'transformer':
            self.transformer_lin_layer.reset_parameters()
            self.transformer_layer.reset_parameters()
        elif self.prediction_type == 'MLP_var3':
            self.orth_sheaf_lin.reset_parameters()
            self.orth_sheaf_lin2.reset_parameters()
        else:
            self.orth_sheaf_lin.reset_parameters()

    def forward(self, x, e, hyperedge_index, debug=False):
        """ 
        x: N x d 
        e: N x f 
        -> (concat) N x E x 2d -> (linear project) N x E x (d*(d-1)//2)
        ->(housholder transform) N x E x (d*(d-1)//2) -> N x E x d x d with each dxd block being an orthonormal matrix
        -> (reshape) (Nd x Ed)

        """
        num_nodes = hyperedge_index[0].max().item() + 1
        num_edges = hyperedge_index[1].max().item() + 1
        x = x.view(num_nodes, self.d, x.shape[-1]).mean(1) # N x d x f -> N x f
        e = e.view(num_edges, self.d, e.shape[-1]).mean(1) # N x d x f -> N x f

        row, col = hyperedge_index
        x_row = torch.index_select(x, dim=0, index=row)
        e_col = torch.index_select(e, dim=0, index=col)

        #Iulia: add layers here instead og layer_idx
        if self.prediction_type == 'transformer':
            h_orth_sheaf = predict_blocks_transformer(x, hyperedge_index, self.transformer_layer, self.transformer_lin_layer, self.args)
        elif self.prediction_type == 'MLP_var1':
            h_orth_sheaf = predict_blocks(x, e, hyperedge_index, self.orth_sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var2':
            h_orth_sheaf = predict_blocks_var2(x, hyperedge_index, self.orth_sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var3':
            h_orth_sheaf = predict_blocks_var3(x, hyperedge_index, self.orth_sheaf_lin, self.orth_sheaf_lin2, self.args)
        
        #convert the d*(d-1)//2 params into orthonormal dxd matrices using housholder transformation
        h_orth_sheaf = self.orth_transform(h_orth_sheaf) #sparse version of a NxExdxd tensor

        if self.sheaf_dropout:
            h_orth_sheaf = F.dropout(h_orth_sheaf, p=self.dropout, training=self.training)
        if self.special_head:
            #add a head having just 1 on the diagonal. this should be similar to the normal hypergraph conv
            #add a head having just 1 on the diagonal. this should be similar to the normal hypergraph conv
            new_head_mask = np.ones((self.d, self.d))
            new_head_mask[:,-1] = np.zeros((self.d))
            new_head_mask[-1,:] = np.zeros((self.d))
            new_head = np.zeros((self.d, self.d))
            new_head[-1,-1] = 1
            h_orth_sheaf = h_orth_sheaf * torch.tensor(new_head_mask, device=x.device) + torch.tensor(new_head, device=x.device)
            h_orth_sheaf = h_orth_sheaf.float()
        # h_orth_sheaf = h_orth_sheaf * torch.eye(self.d, device=self.device)

        # from a d-dim tensor assoc to every entrence in edge_inde
        # create a sparse incidence Nd x Ed
        # modify indices to correspond to the big matrix and assign the weights
        # indices: [i,j] -> [d*i, d*i.. d*i+d-1, d*i+d-1; d*j, d*j+1 .. d*j, d*j+1,..d*j+d-1]

        if (debug==True):
            print("x", x.mean(-1))
            print("x_row", x_row.mean(-1))
            print("e", e.mean(-1))
            print("e_col", e_col.mean(-1))
            print(hyperedge_index)

        d_range = torch.arange(self.d, device=x.device)
        d_range_edges = d_range.repeat(self.d).view(-1,1) #0,1..d,0,1..d..   d*d elems
        d_range_nodes = d_range.repeat_interleave(self.d).view(-1,1) #0,0..0,1,1..1..d,d..d  d*d elems
        hyperedge_index = hyperedge_index.unsqueeze(1) 
   

        hyperedge_index_0 = self.d * hyperedge_index[0] + d_range_nodes
        hyperedge_index_0 = hyperedge_index_0.permute((1,0)).reshape(1,-1)
        hyperedge_index_1 = self.d * hyperedge_index[1] + d_range_edges
        hyperedge_index_1 = hyperedge_index_1.permute((1,0)).reshape(1,-1)
        h_orth_sheaf_index = torch.concat((hyperedge_index_0, hyperedge_index_1), 0)
        #!!! Is this the correct reshape??? Please check!!
        h_orth_sheaf_attributes = h_orth_sheaf.reshape(-1)
        
        #create the big matrix from the dxd orthogonal blocks  
        return h_orth_sheaf_index, h_orth_sheaf_attributes


#functions used to predict the blocks

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

def predict_blocks_var3(x, hyperedge_index, sheaf_lin, sheaf_lin2, args):
    # universal approx according to  Equivariant Hypergraph Diffusion Neural Operators
    # ρ(zi, sum(φ(zj))

    #here hypergraph_att  is always the average of the intermediate x features
    row, col = hyperedge_index
    #φ(zj)
    x_e = sheaf_lin2(x)
    #sum(φ(zj)
    e = scatter_add(x_e[row],col, dim=0)
    
    xs = torch.index_select(x, dim=0, index=row)
    es= torch.index_select(e, dim=0, index=col)


    # select all pairs (zi, sum(φ(zj))
    h_sheaf = torch.cat((xs,es), dim=-1) #sparse v ersion of an NxEx2f tensor

    # ρ(zi || sum(φ(zj))
    h_sheaf = sheaf_lin(h_sheaf)  #sparse version of an NxExd tensor


    if args.sheaf_act == 'sigmoid':
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



class HGCNSheafBuilderDiag(nn.Module):
    def __init__(self, args, hidden_dim):
        """
        hidden_dim overwrite the args.MLP_hidden used in the normal sheaf HNN
        """
        super(HGCNSheafBuilderDiag, self).__init__()
        self.args = args
        self.prediction_type = args.sheaf_pred_block
        self.sheaf_dropout = args.sheaf_dropout
        self.special_head = args.sheaf_special_head
        self.d = args.heads
        self.MLP_hidden = hidden_dim
        self.norm = args.AllSet_input_norm

        if self.prediction_type == 'transformer':
            transformer_head = args.sheaf_transformer_head
            self.transformer_layer=torch_geometric.nn.TransformerConv(
                                in_channels=self.MLP_hidden, 
                                out_channels=self.MLP_hidden//transformer_head,
                                heads=transformer_head)
                                
            self.transformer_lin_layer = MLP(
                        in_channels=2*self.MLP_hidden, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=self.d,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
                        
        elif self.prediction_type == 'MLP_var1':
            self.sheaf_lin = MLP(
                        in_channels=self.MLP_hidden + args.MLP_hidden,  
                        hidden_channels=args.MLP_hidden,
                        out_channels=self.d,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
        elif self.prediction_type in ['MLP_var2', 'MLP_var3']:
            self.sheaf_lin = MLP(
                        in_channels=2*self.MLP_hidden,  
                        hidden_channels=args.MLP_hidden,
                        out_channels=self.d,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
        if self.prediction_type == 'MLP_var3':
            self.sheaf_lin2 = MLP(
                        in_channels=self.MLP_hidden, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=self.MLP_hidden,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
                        

    def reset_parameters(self):
        if self.prediction_type == 'transformer':
            self.transformer_lin_layer.reset_parameters()
            self.transformer_layer.reset_parameters()
        elif self.prediction_type == 'MLP_var3':
            self.sheaf_lin.reset_parameters()
            self.sheaf_lin2.reset_parameters()
        else:
            self.sheaf_lin.reset_parameters()
        

    #this is exclusively for diagonal sheaf
    def forward(self, x, e, hyperedge_index):
        """ tmp
        x: Nd x f -> N x f
        e: Ed x f -> E x f
        -> (concat) N x E x (d+1)F -> (linear project) N x E x d (the elements on the diagonal of each dxd block)
        -> (reshape) (Nd x Ed) with NxE diagonal blocks of dimension dxd

        """

        num_nodes = hyperedge_index[0].max().item() + 1
        num_edges = hyperedge_index[1].max().item() + 1
        x = x.view(num_nodes, self.d, x.shape[-1]).mean(1) # N x d x f -> N x f
        e = e.view(num_edges, self.d, e.shape[-1]).mean(1) # # x d x f -> E x f

        # h_sheaf = self.predict_blocks(x, e, hyperedge_index, sheaf_lin)
        # h_sheaf = self.predict_blocks_var2(x, hyperedge_index, sheaf_lin)
        if self.prediction_type == 'transformer':
            h_sheaf = predict_blocks_transformer(x, hyperedge_index, self.transformer_layer, self.transformer_lin_layer, self.args)
        elif self.prediction_type == 'MLP_var1':
            h_sheaf = predict_blocks(x, e, hyperedge_index, self.sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var2':
            h_sheaf = predict_blocks_var2(x, hyperedge_index, self.sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var3':
            h_sheaf = predict_blocks_var3(x, hyperedge_index, self.sheaf_lin,  self.sheaf_lin2, self.args)

        if self.sheaf_dropout:
            h_sheaf = F.dropout(h_sheaf, p=self.dropout, training=self.training)
        #add a head having just 1 on the diagonal. this should be similar to the normal hypergraph conv
        if self.special_head:
            new_head_mask = [1]*(self.d-1) + [0]
            new_head = [0]*(self.d-1) + [1]
            h_sheaf = h_sheaf * torch.tensor(new_head_mask, device=x.device) + torch.tensor(new_head, device=x.device)
        
        # h_sheaf = torch.diag_embed(h_sheaf, dim1=-2, dim2=-1)
        return h_sheaf
        # self.h_sheaf = h_sheaf #this is stored in self for testing purpose

class HGCNSheafBuilderGeneral(nn.Module):
    def __init__(self, args, hidden_dim):
        super(HGCNSheafBuilderGeneral, self).__init__()
        self.args = args
        self.prediction_type = args.sheaf_pred_block
        self.sheaf_dropout = args.sheaf_dropout
        self.special_head = args.sheaf_special_head
        self.d = args.heads
        self.MLP_hidden = hidden_dim
        self.norm = args.AllSet_input_norm

        
        if self.prediction_type == 'transformer':
            transformer_head = args.sheaf_transformer_head
            self.transformer_layer=torch_geometric.nn.TransformerConv(
                                in_channels=self.MLP_hidden, 
                                out_channels=self.MLP_hidden//transformer_head,
                                heads=transformer_head)
                                
            self.transformer_lin_layer = MLP(
                        in_channels=2*self.MLP_hidden, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=self.d*self.d,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
                        
        elif self.prediction_type == 'MLP_var1':
            self.sheaf_lin = MLP(
                        in_channels=self.MLP_hidden + args.MLP_hidden, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=self.d*self.d,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
        elif self.prediction_type in ['MLP_var2', 'MLP_var3']:
            self.sheaf_lin = MLP(
                        in_channels=2*self.MLP_hidden, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=self.d*self.d,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
        if self.prediction_type == 'MLP_var3':
            self.sheaf_lin2 = MLP(
                        in_channels=self.MLP_hidden, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=self.MLP_hidden,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
                        

    def reset_parameters(self):
        if self.prediction_type == 'transformer':
            self.transformer_lin_layer.reset_parameters()
            self.transformer_layer.reset_parameters()
        elif self.prediction_type == 'MLP_var3':
            self.sheaf_lin.reset_parameters()
            self.sheaf_lin2.reset_parameters()
        else:
            self.sheaf_lin.reset_parameters()
        

    #this is exclusively for diagonal sheaf
    def forward(self, x, e, hyperedge_index):
        """ tmp
        x: Nd x f -> N x f
        e: Ed x f -> E x f
        -> (concat) N x E x (d+1)F -> (linear project) N x E x d (the elements on the diagonal of each dxd block)
        -> (reshape) (Nd x Ed) with NxE diagonal blocks of dimension dxd

        """

        num_nodes = hyperedge_index[0].max().item() + 1
        num_edges = hyperedge_index[1].max().item() + 1
        x = x.view(num_nodes, self.d, x.shape[-1]).mean(1) # N x d x f -> N x f
        e = e.view(num_edges, self.d, e.shape[-1]).mean(1) # # x d x f -> E x f

        # h_sheaf = self.predict_blocks(x, e, hyperedge_index, sheaf_lin)
        # h_sheaf = self.predict_blocks_var2(x, hyperedge_index, sheaf_lin)
        if self.prediction_type == 'transformer':
            h_sheaf = predict_blocks_transformer(x, hyperedge_index, self.transformer_layer, self.transformer_lin_layer, self.args)
        elif self.prediction_type == 'MLP_var1':
            h_sheaf = predict_blocks(x, e, hyperedge_index, self.sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var2':
            h_sheaf = predict_blocks_var2(x, hyperedge_index, self.sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var3':
            h_sheaf = predict_blocks_var3(x, hyperedge_index, self.sheaf_lin,  self.sheaf_lin2, self.args)

        if self.sheaf_dropout:
            h_sheaf = F.dropout(h_sheaf, p=self.dropout, training=self.training)
        #add a head having just 1 on the diagonal. this should be similar to the normal hypergraph conv
        
        # h_sheaf = torch.diag_embed(h_sheaf, dim1=-2, dim2=-1)
        return h_sheaf
        # self.h_sheaf = h_sheaf #this is stored in self for testing purpose


class HGCNSheafBuilderOrtho(nn.Module):
    def __init__(self, args, hidden_dim):
        super(HGCNSheafBuilderOrtho, self).__init__()
        self.args = args
        self.prediction_type = args.sheaf_pred_block
        self.sheaf_dropout = args.sheaf_dropout
        self.special_head = args.sheaf_special_head
        self.d = args.heads
        self.MLP_hidden = hidden_dim
        self.norm = args.AllSet_input_norm

        self.orth_transform = Orthogonal(d=self.d, orthogonal_map='householder') #method applied to transform params into ortho dxd matrix
        
        if self.prediction_type == 'transformer':
            transformer_head = args.sheaf_transformer_head
            self.transformer_layer=torch_geometric.nn.TransformerConv(
                                in_channels=self.MLP_hidden, 
                                out_channels=self.MLP_hidden//transformer_head,
                                heads=transformer_head)
                                
            self.transformer_lin_layer = MLP(
                        in_channels=2*self.MLP_hidden, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=self.d*(self.d-1)//2,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
                        
        elif self.prediction_type == 'MLP_var1':
            self.sheaf_lin = MLP(
                        in_channels=self.MLP_hidden + args.MLP_hidden, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=self.d*(self.d-1)//2,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
        elif self.prediction_type in ['MLP_var2', 'MLP_var3']:
            self.sheaf_lin = MLP(
                        in_channels=2*self.MLP_hidden, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=self.d*(self.d-1)//2,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
        if self.prediction_type == 'MLP_var3':
            self.sheaf_lin2 = MLP(
                        in_channels=self.MLP_hidden, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=self.MLP_hidden,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
                        

    def reset_parameters(self):
        if self.prediction_type == 'transformer':
            self.transformer_lin_layer.reset_parameters()
            self.transformer_layer.reset_parameters()
        elif self.prediction_type == 'MLP_var3':
            self.sheaf_lin.reset_parameters()
            self.sheaf_lin2.reset_parameters()
        else:
            self.sheaf_lin.reset_parameters()
        

    #this is exclusively for diagonal sheaf
    def forward(self, x, e, hyperedge_index):
        """ tmp
        x: Nd x f -> N x f
        e: Ed x f -> E x f
        -> (concat) N x E x (d+1)F -> (linear project) N x E x d (the elements on the diagonal of each dxd block)
        -> (reshape) (Nd x Ed) with NxE diagonal blocks of dimension dxd

        """

        num_nodes = hyperedge_index[0].max().item() + 1
        num_edges = hyperedge_index[1].max().item() + 1
        x = x.view(num_nodes, self.d, x.shape[-1]).mean(1) # N x d x f -> N x f
        e = e.view(num_edges, self.d, e.shape[-1]).mean(1) # # x d x f -> E x f

        # h_sheaf = self.predict_blocks(x, e, hyperedge_index, sheaf_lin)
        # h_sheaf = self.predict_blocks_var2(x, hyperedge_index, sheaf_lin)
        if self.prediction_type == 'transformer':
            h_sheaf = predict_blocks_transformer(x, hyperedge_index, self.transformer_layer, self.transformer_lin_layer, self.args)
        elif self.prediction_type == 'MLP_var1':
            h_sheaf = predict_blocks(x, e, hyperedge_index, self.sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var2':
            h_sheaf = predict_blocks_var2(x, hyperedge_index, self.sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var3':
            h_sheaf = predict_blocks_var3(x, hyperedge_index, self.sheaf_lin,  self.sheaf_lin2, self.args)

        h_sheaf = self.orth_transform(h_sheaf) #sparse version of a NxExdxd tensor
        if self.sheaf_dropout:
            h_sheaf = F.dropout(h_sheaf, p=self.dropout, training=self.training)
        #add a head having just 1 on the diagonal. this should be similar to the normal hypergraph conv
        
        # h_sheaf = torch.diag_embed(h_sheaf, dim1=-2, dim2=-1)
        return h_sheaf
        # self.h_sheaf = h_sheaf #this is stored in self for testing purpose