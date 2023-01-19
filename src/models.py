#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 
#
# Distributed under terms of the MIT license.

"""
This script contains all models in our paper.
"""

import torch
import utils

import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing, GCNConv, GATConv
from layers import *


import math 

from torch_scatter import scatter, scatter_mean, scatter_add
from torch_geometric.utils import softmax
import pdb

from orthogonal import Orthogonal
#  This part is for HyperGCN

class HyperGCN(nn.Module):
    def __init__(self, V, E, X, num_features, num_layers, num_classses, args):
        """
        d: initial node-feature dimension
        h: number of hidden units
        c: number of classes
        """
        super(HyperGCN, self).__init__()
        d, l, c = num_features, num_layers, num_classses
        cuda = args.cuda  # and torch.cuda.is_available()

        h = [d]
        for i in range(l-1):
            power = l - i + 2
            if args.dname == 'citeseer':
                power = l - i + 4
            h.append(2**power)
        h.append(c)

        if args.HyperGCN_fast:
            reapproximate = False
            structure = utils.Laplacian(V, E, X, args.HyperGCN_mediators)
        else:
            reapproximate = True
            structure = E

        self.layers = nn.ModuleList([utils.HyperGraphConvolution(
            h[i], h[i+1], reapproximate, cuda) for i in range(l)])
        self.do, self.l = args.dropout, num_layers
        self.structure, self.m = structure, args.HyperGCN_mediators

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data):
        """
        an l-layer GCN
        """
        do, l, m = self.do, self.l, self.m
        H = data.x

        for i, hidden in enumerate(self.layers):
            H = F.relu(hidden(self.structure, H, m))
            if i < l - 1:
                V = H
                H = F.dropout(H, do, training=self.training)

        return H


class CEGCN(MessagePassing):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 num_layers,
                 dropout,
                 Normalization='bn'
                 ):
        super(CEGCN, self).__init__()
        self.convs = nn.ModuleList()
        self.normalizations = nn.ModuleList()

        if Normalization == 'bn':
            self.convs.append(GCNConv(in_dim, hid_dim, normalize=False))
            self.normalizations.append(nn.BatchNorm1d(hid_dim))
            for _ in range(num_layers-2):
                self.convs.append(GCNConv(hid_dim, hid_dim, normalize=False))
                self.normalizations.append(nn.BatchNorm1d(hid_dim))

            self.convs.append(GCNConv(hid_dim, out_dim, normalize=False))
        else:  # default no normalizations
            self.convs.append(GCNConv(in_dim, hid_dim, normalize=False))
            self.normalizations.append(nn.Identity())
            for _ in range(num_layers-2):
                self.convs.append(GCNConv(hid_dim, hid_dim, normalize=False))
                self.normalizations.append(nn.Identity())

            self.convs.append(GCNConv(hid_dim, out_dim, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for layer in self.convs:
            layer.reset_parameters()
        for normalization in self.normalizations:
            if not (normalization.__class__.__name__ is 'Identity'):
                normalization.reset_parameters()

    def forward(self, data):
        #         Assume edge_index is already V2V
        x, edge_index, norm = data.x, data.edge_index, data.norm
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, norm)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, norm)
        return x


class CEGAT(MessagePassing):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 num_layers,
                 heads,
                 output_heads,
                 dropout,
                 Normalization='bn'
                 ):
        super(CEGAT, self).__init__()
        self.convs = nn.ModuleList()
        self.normalizations = nn.ModuleList()

        if Normalization == 'bn':
            self.convs.append(GATConv(in_dim, hid_dim, heads))
            self.normalizations.append(nn.BatchNorm1d(hid_dim))
            for _ in range(num_layers-2):
                self.convs.append(GATConv(heads*hid_dim, hid_dim))
                self.normalizations.append(nn.BatchNorm1d(hid_dim))

            self.convs.append(GATConv(heads*hid_dim, out_dim,
                                      heads=output_heads, concat=False))
        else:  # default no normalizations
            self.convs.append(GATConv(in_dim, hid_dim, heads))
            self.normalizations.append(nn.Identity())
            for _ in range(num_layers-2):
                self.convs.append(GATConv(hid_dim*heads, hid_dim))
                self.normalizations.append(nn.Identity())

            self.convs.append(GATConv(hid_dim*heads, out_dim,
                                      heads=output_heads, concat=False))

        self.dropout = dropout

    def reset_parameters(self):
        for layer in self.convs:
            layer.reset_parameters()
        for normalization in self.normalizations:
            if not (normalization.__class__.__name__ is 'Identity'):
                normalization.reset_parameters()

    def forward(self, data):
        #         Assume edge_index is already V2V
        x, edge_index, norm = data.x, data.edge_index, data.norm
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)

    def reset_parameters(self):
        self.hgc1.reset_parameters()
        self.hgc2.reset_parameters()

    def forward(self, data):
        x = data.x
        G = data.edge_index

        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        return x

class DiagSheafs(nn.Module):
    """
        This is a Hypergraph Sheaf Model with 
        the dxd blocks in H_BIG associated to each pair (node, hyperedge)
        being **diagonal**


    """
    def __init__(self, args):
        super(DiagSheafs, self).__init__()

        self.num_layers = args.All_num_layers
        self.dropout = args.dropout  # Note that default is 0.6
        self.num_features = args.num_features
        self.MLP_hidden = args.MLP_hidden 
        self.d = args.heads # dimension of the stalks
        self.init_hedge = args.init_hedge # how to initialise hyperedge attributes: avg or rand
        self.norm_type = args.sheaf_normtype #type of laplacian normalisation degree_norm or block_norm
        self.act = args.sheaf_act # type of nonlinearity used when predicting the dxd blocks
        self.hyperedge_attr = None
        self.sheaf_dropout = args.sheaf_dropout #dropout used/not-used in predicting the dxd blocks
        self.left_proj = args.sheaf_left_proj
        if args.cuda in [0, 1]:
            self.device = torch.device('cuda:'+str(args.cuda)
                              if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        self.lin = Linear(self.num_features, self.num_features*self.d, bias=False)

        self.dynamic_sheaf = args.dynamic_sheaf
        self.sheaf_lin = nn.ModuleList()
#         Note that add dropout to attention is default in the original paper
        self.convs = nn.ModuleList()
        self.convs.append(HypergraphDiagSheafConv(self.num_features, self.MLP_hidden, d=self.d, device=self.device, norm_type=self.norm_type, left_proj=self.left_proj))
        self.sheaf_lin.append(Linear(2*self.num_features, self.d, bias=False))
        
        #iulia Qs: add back the multi-layers?
        for _ in range(self.num_layers-1):
            self.convs.append(HypergraphDiagSheafConv(self.MLP_hidden, self.MLP_hidden, d=self.d, device=self.device, norm_type=self.norm_type, left_proj=self.left_proj))
            if self.dynamic_sheaf:
                self.sheaf_lin.append(Linear(self.MLP_hidden+self.num_features, self.d, bias=False))

        self.lin2 = Linear(self.MLP_hidden*self.d, args.num_classes, bias=False)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for sheaf_lin in self.sheaf_lin:
            sheaf_lin.reset_parameters()
        self.lin.reset_parameters()
        self.lin2.reset_parameters()
        

    def predict_blocks(self, xs, es, sheaf_lin):
        # select all pairs (node, hyperedge)
        h_sheaf = torch.cat((xs,es), dim=-1) #sparse version of an NxEx2f tensor
        h_sheaf = sheaf_lin(h_sheaf)  #sparse version of an NxExd tensor
        if self.act == 'sigmoid':
            h_sheaf = F.sigmoid(h_sheaf) # output d numbers for every entry in the incidence matrix
        elif self.act == 'tanh':
            h_sheaf = F.tanh(h_sheaf) # output d numbers for every entry in the incidence matrix
        
        if self.sheaf_dropout:
            h_sheaf = F.dropout(h_sheaf, p=self.dropout, training=self.training)
        return h_sheaf

    #this is exclusively for diagonal sheaf
    def build_sheaf_incidence(self, x, e, hyperedge_index, sheaf_lin):
        """ tmp
        x: Nd x f -> N x f
        e: Ed x f -> E x f
        -> (concat) N x E x (d+1)F -> (linear project) N x E x d (the elements on the diagonal of each dxd block)
        -> (reshape) (Nd x Ed) with NxE diagonal blocks of dimension dxd

        """
        num_nodes = hyperedge_index[0].max().item() + 1
        num_edges = hyperedge_index[1].max().item() + 1
        x = x.view(num_nodes, self.d, x.shape[-1]).mean(1) # N x d x f -> N x f
        e = e.view(num_edges, self.d, e.shape[-1]).mean(1) # N x d x f -> N x f


        row, col = hyperedge_index

        x_row = torch.index_select(x, dim=0, index=row)
        e_col = torch.index_select(e, dim=0, index=col)
        h_sheaf = self.predict_blocks(x_row, e_col, sheaf_lin)
        
        self.h_sheaf = h_sheaf #this is stored in self for testing purpose

        # from a d-dim tensor assoc to every entrence in edge_index
        # create a sparse incidence Nd x Ed

        # We need to modify indices from the NxE matrix 
        # to correspond to the large Nd x Ed matrix, but restrict only on the element of the diagonal of each block
        # indices: scalar [i,j] -> block dxd with indices [d*i, d*i+1.. d*i+d-1; d*j, d*j+1 .. d*j+d-1]
        # attributes: reshape h_sheaf

        d_range = torch.arange(self.d, device=self.device).view(1,-1,1).repeat(2,1,1) #2xdx1
        hyperedge_index = hyperedge_index.unsqueeze(1) #2x1xK
        hyperedge_index = self.d * hyperedge_index + d_range #2xdxK

        hyperedge_index = hyperedge_index.permute(0,2,1).reshape(2,-1) #2x(d*K)

        h_sheaf_index = hyperedge_index
        h_sheaf_attributes = h_sheaf.reshape(-1) #(d*K)

        #the resulting (index, values) pair correspond to the diagonal of each block sub-matrix
        return h_sheaf_index, h_sheaf_attributes

    def init_hyperedge_attr(self, type, num_edges=None, x=None, hyperedge_index=None):
        #initialize hyperedge attributes either random or as the average of the nodes
        if type == 'rand':
            hyperedge_attr = torch.randn((num_edges, self.num_features)).to(self.device)
        elif type == 'avg':
            hyperedge_attr = scatter_mean(x[hyperedge_index[0]],hyperedge_index[1], dim=0)
        else:
            hyperedge_attr = None
        return hyperedge_attr

    def forward(self, data):

        x = data.x
        edge_index = data.edge_index
        num_nodes = data.edge_index[0].max().item() + 1
        num_edges = data.edge_index[1].max().item() + 1

        # hyperedge_attr = torch.randn((num_edges, self.num_features)).to(self.device)

        #if we are at the first epoch, initialise the attribute, otherwise use the previous ones
        if self.hyperedge_attr is None:
            self.hyperedge_attr = self.init_hyperedge_attr(self.init_hedge, num_edges=num_edges, x=x, hyperedge_index=edge_index)
        
        # #infer the sheaf as a sparse incidence matrix Nd x Ed, with each block being diagonal
        # h_sheaf_index, h_sheaf_attributes = self.build_sheaf_incidence(x, self.hyperedge_attr, edge_index)

        # expand the input N x num_features -> Nd x num_features such that we can apply the propagation
        x = self.lin(x)
        hyperedge_attr = self.lin(self.hyperedge_attr)

        x = x.view((x.shape[0]*self.d, self.num_features)) # (N * d) x num_features
        hyperedge_attr = hyperedge_attr.view((hyperedge_attr.shape[0]*self.d, self.num_features))

        for i, conv in enumerate(self.convs[:-1]):
            #infer the sheaf as a sparse incidence matrix Nd x Ed, with each block being diagonal
            if i == 0 or self.dynamic_sheaf:
                h_sheaf_index, h_sheaf_attributes = self.build_sheaf_incidence(x, hyperedge_attr, edge_index, self.sheaf_lin[i])
            x = F.elu(conv(x, hyperedge_index=h_sheaf_index, alpha=h_sheaf_attributes, num_nodes=num_nodes, num_edges=num_edges))
            x = F.dropout(x, p=self.dropout, training=self.training)


        if self.dynamic_sheaf:
            h_sheaf_index, h_sheaf_attributes = self.build_sheaf_incidence(x, hyperedge_attr, edge_index, self.sheaf_lin[-1])
        x = self.convs[-1](x,  hyperedge_index=h_sheaf_index, alpha=h_sheaf_attributes, num_nodes=num_nodes, num_edges=num_edges)
        x = x.view(num_nodes, -1) # Nd x out_channels -> Nx(d*out_channels)
        x = self.lin2(x) # Nx(d*out_channels)-> N x num_classes

        return x


class OrthoSheafs(nn.Module):
    """
        This is a Hypergraph Sheaf Model with 
        the dxd blocks in H associated to each pair (node, hyperedge)
        being **othogonal**


    """
    def __init__(self, args):
        super(OrthoSheafs, self).__init__()

        self.num_layers = args.All_num_layers
        self.dropout = args.dropout  # Note that default is 0.6
        self.num_features = args.num_features
        self.MLP_hidden = args.MLP_hidden 
        self.d = args.heads # dimension of the stalks
        self.init_hedge = args.init_hedge # how to initialise hyperedge attributes: avg or rand
        self.norm_type = args.sheaf_normtype #type of laplacian normalisation degree_norm or block_norm
        self.act = args.sheaf_act # type of nonlinearity used when predicting the dxd blocks
        self.hyperedge_attr = None
        self.sheaf_dropout = args.sheaf_dropout #dropout used/not-used in predicting the dxd blocks
        self.left_proj = args.sheaf_left_proj

        if args.cuda in [0, 1]:
            self.device = torch.device('cuda:'+str(args.cuda)
                              if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        self.dynamic_sheaf = args.dynamic_sheaf
        self.orth_sheaf_lin = nn.ModuleList()

        self.lin = Linear(args.num_features, args.num_features*self.d, bias=False)
        self.orth_sheaf_lin.append(Linear(2*args.num_features, self.d*(self.d-1)//2, bias=False)) #d(d-1)/2 params to transform in an ortho matrix
        self.orth_transform = Orthogonal(d=self.d, orthogonal_map='householder') #method applied to transform params into ortho dxd matrix

#       Note that add dropout to attention is default in the original paper
        self.convs = nn.ModuleList()
        self.convs.append(HypergraphOrthoSheafConv(args.num_features, self.MLP_hidden, d=self.d, device=self.device, norm_type=self.norm_type, left_proj=self.left_proj))
        
        for _ in range(self.num_layers-1):
            self.convs.append(HypergraphOrthoSheafConv(args.MLP_hidden, args.MLP_hidden, d=self.d, device=self.device, norm_type=self.norm_type, left_proj=self.left_proj))
            if self.dynamic_sheaf:
                self.orth_sheaf_lin.append(Linear(args.num_features+args.MLP_hidden, self.d*(self.d-1)//2, bias=False)) 
        self.lin2 = Linear(self.MLP_hidden*self.d, args.num_classes, bias=False)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for ortho_sheaf_lin in self.orth_sheaf_lin:
            ortho_sheaf_lin.reset_parameters()
        self.lin.reset_parameters()
        self.lin2.reset_parameters()
        

    def predict_blocks(self, xs, es, orth_sheaf_lin):
        
        # select all pairs (node, hyperedge)
        h_sheaf = torch.cat((xs,es), dim=-1)  #sparse version of a NxEx2f tensor
        h_sheaf = orth_sheaf_lin(h_sheaf)  #output d(d-1)//2 numbers for every entry in the incidence matrix
        
        if self.act == 'sigmoid':
            h_sheaf = F.sigmoid(h_sheaf)
        elif self.act == 'tanh':
            h_sheaf = F.tanh(h_sheaf)

        #convert the d*(d-1)//2 params into orthonormal dxd matrices using housholder transformation
        h_orth_sheaf = self.orth_transform(h_sheaf) #sparse version of a NxExdxd tensor
        if self.sheaf_dropout:
            h_orth_sheaf = F.dropout(h_orth_sheaf, p=self.dropout, training=self.training)
        return h_orth_sheaf

    def build_ortho_sheaf_incidence(self, x, e, hyperedge_index, orth_sheaf_lin, debug=False):
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
        h_orth_sheaf = self.predict_blocks(x_row, e_col, orth_sheaf_lin)
        
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

        d_range = torch.arange(self.d, device=self.device)
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

    def init_hyperedge_attr(self, type, num_edges=None, x=None, hyperedge_index=None):
        if type == 'rand':
            hyperedge_attr = torch.randn((num_edges, self.num_features)).to(self.device)
        elif type == 'avg':
            hyperedge_attr = scatter_mean(x[hyperedge_index[0]],hyperedge_index[1], dim=0)
        else:
            hyperedge_attr = None
        return hyperedge_attr

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        num_nodes = data.edge_index[0].max().item() + 1
        num_edges = data.edge_index[1].max().item() + 1

        # hyperedge_attr = torch.randn((num_edges, self.num_features)).to(self.device)

        #if we are at the first epoch, initialise the attribute, otherwise use the previous ones
        if self.hyperedge_attr is None:
            self.hyperedge_attr = self.init_hyperedge_attr(self.init_hedge, num_edges=num_edges, x=x, hyperedge_index=edge_index)

        #infer the sheaf as a sparse incidence matrix Nd x Ed
        # h_orth_sheaf_index, h_orth_sheaf_attributes = self.build_ortho_sheaf_incidence(x, self.hyperedge_attr, edge_index, self.orth_sheaf_lin[i])
        #expand the input Nd x num_features
        x = self.lin(x) #N x num_features -> N x (d*num_features)
        hyperedge_attr = self.lin(self.hyperedge_attr)
        x = x.view((x.shape[0]*self.d, self.num_features)) # (N * d) x num_features
        hyperedge_attr = hyperedge_attr.view((hyperedge_attr.shape[0]*self.d, self.num_features)) # (E * d) x num_features
        
        for i, conv in enumerate(self.convs[:-1]):
            if i == 0 or self.dynamic_sheaf:
                h_orth_sheaf_index, h_orth_sheaf_attributes = self.build_ortho_sheaf_incidence(x, hyperedge_attr, edge_index, self.orth_sheaf_lin[i])
            x = F.elu(conv(x, hyperedge_index=h_orth_sheaf_index, alpha=h_orth_sheaf_attributes, num_nodes=num_nodes, num_edges=num_edges))
            x = F.dropout(x, p=self.dropout, training=self.training)
#         x = F.dropout(x, p=self.dropout, training=self.training)

        if self.dynamic_sheaf:
            h_orth_sheaf_index, h_orth_sheaf_attributes = self.build_ortho_sheaf_incidence(x, hyperedge_attr, edge_index, self.orth_sheaf_lin[-1])
        x = self.convs[-1](x,  hyperedge_index=h_orth_sheaf_index, alpha=h_orth_sheaf_attributes, num_nodes=num_nodes, num_edges=num_edges)
        x = x.view(num_nodes, -1) #Nd x out_channels -> Nx(d*out_channels)
        x = self.lin2(x) #N x (d*out_channels) -> N x num_classes
        return x




class GeneralSheafs(nn.Module):
    """
        This is a Hypergraph Sheaf Model with 
        the dxd blocks in H associated to each pair (node, hyperedge)
        being **unconstrained**


    """
    def __init__(self, args):
        super(GeneralSheafs, self).__init__()

        self.num_layers = args.All_num_layers
        self.dropout = args.dropout  # Note that default is 0.6
        self.num_features = args.num_features
        self.MLP_hidden = args.MLP_hidden 
        self.d = args.heads # dimension of the stalks
        self.init_hedge = args.init_hedge # how to initialise hyperedge attributes: avg or rand
        self.norm_type = args.sheaf_normtype #type of laplacian normalisation degree_norm or block_norm
        assert self.norm_type == 'degree_norm' #block_norm still has bugss
        self.act = args.sheaf_act # type of nonlinearity used when predicting the dxd blocks
        self.hyperedge_attr = None
        self.sheaf_dropout = args.sheaf_dropout #dropout used/not-used in predicting the dxd blocks
        self.left_proj = args.sheaf_left_proj

        if args.cuda in [0, 1]:
            self.device = torch.device('cuda:'+str(args.cuda)
                              if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        self.dynamic_sheaf = args.dynamic_sheaf
        self.general_sheaf_lin = nn.ModuleList()

        self.lin = Linear(args.num_features, args.num_features*self.d, bias=False)
        self.general_sheaf_lin.append(Linear(2*args.num_features, self.d*self.d, bias=False)) #d(d-1)/2 params to transform in an ortho matrix

#         Note that add dropout to attention is default in the original paper
        self.convs = nn.ModuleList()
        self.convs.append(HypergraphGeneralSheafConv(args.num_features, self.MLP_hidden, d=self.d, device=self.device, norm_type=self.norm_type, left_proj=self.left_proj))
        #iulia Qs: add back the multi-layers?
        for _ in range(self.num_layers-1):
            if self.dynamic_sheaf:
                self.general_sheaf_lin.append(Linear(args.num_features+args.MLP_hidden, self.d*self.d, bias=False))
            self.convs.append(HypergraphGeneralSheafConv(args.MLP_hidden, args.MLP_hidden, d=self.d, device=self.device, norm_type=self.norm_type, left_proj=self.left_proj))

        self.lin2 = Linear(self.MLP_hidden*self.d, args.num_classes, bias=False)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for general_sheaf_lin in self.general_sheaf_lin:
            general_sheaf_lin.reset_parameters()
        self.lin.reset_parameters()
        self.lin2.reset_parameters()

    def predict_blocks(self, xs, es, general_sheaf_lin, type='concat_lin'):
        # select each pair (node, hedge)
        h_sheaf = torch.cat((xs,es), dim=-1) 
        h_sheaf = general_sheaf_lin(h_sheaf)  #output d*d numbers for every entry in the incidence matrix
        if self.act == 'sigmoid':
            h_sheaf = F.sigmoid(h_sheaf)
        elif self.act == 'tanh':
            h_sheaf = F.tanh(h_sheaf)
        if self.sheaf_dropout:
            h_sheaf = F.dropout(h_sheaf, p=self.dropout, training=self.training)
        h_sheaf = h_sheaf.view(h_sheaf.shape[0], self.d, self.d) #dxd ortho matrix
        return h_sheaf
         
    def build_general_sheaf_incidence(self, x, e, hyperedge_index, general_sheaf_lin, debug=False):
        """ 
        x: N x f
        e: N x f 
        -> (concat) N x E x 2f -> (linear project) N x E x d*d
        -> (reshape) (Nd x Ed) with each block dxd being unconstrained

        """
        row, col = hyperedge_index
        x_row = torch.index_select(x, dim=0, index=row)
        e_col = torch.index_select(e, dim=0, index=col)

        h_general_sheaf = self.predict_blocks(x_row, e_col, general_sheaf_lin)
        #Iulia: Temporary debug
        # h_general_sheaf = h_general_sheaf * torch.eye(self.d, device=self.device)
        self.h_general_sheaf = h_general_sheaf #for debug purpose

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
        
        d_range = torch.arange(self.d, device=self.device)
        d_range_edges = d_range.repeat(self.d).view(-1,1) #0,1..d,0,1..d..   d*d elems
        d_range_nodes = d_range.repeat_interleave(self.d).view(-1,1) #0,0..0,1,1..1..d,d..d  d*d elems
        hyperedge_index = hyperedge_index.unsqueeze(1) 
   

        hyperedge_index_0 = self.d * hyperedge_index[0] + d_range_nodes
        hyperedge_index_0 = hyperedge_index_0.permute((1,0)).reshape(1,-1)
        hyperedge_index_1 = self.d * hyperedge_index[1] + d_range_edges
        hyperedge_index_1 = hyperedge_index_1.permute((1,0)).reshape(1,-1)
        h_general_sheaf_index = torch.concat((hyperedge_index_0, hyperedge_index_1), 0)

        if self.norm_type == 'block_norm':
            pass
            # num_nodes = hyperedge_index[0].max().item() + 1
            # num_edges = hyperedge_index[1].max().item() + 1

            # to_be_inv_nodes = torch.bmm(h_general_sheaf, h_general_sheaf.permute(0,2,1)) 
            # to_be_inv_nodes = scatter_add(to_be_inv_nodes, row, dim=0, dim_size=num_nodes)

            # to_be_inv_edges = torch.bmm(h_general_sheaf.permute(0,2,1), h_general_sheaf)
            # to_be_inv_edges = scatter_add(to_be_inv_edges, col, dim=0, dim_size=num_edges)


            # d_sqrt_inv_nodes = utils.batched_sym_matrix_pow(to_be_inv_nodes, -1.0) #n_nodes x d x d
            # d_sqrt_inv_edges = utils.batched_sym_matrix_pow(to_be_inv_edges, -1.0) #n_edges x d x d
            

            # d_sqrt_inv_nodes_large = torch.index_select(d_sqrt_inv_nodes, dim=0, index=row)
            # d_sqrt_inv_edges_large = torch.index_select(d_sqrt_inv_edges, dim=0, index=col)


            # alpha_norm = torch.bmm(d_sqrt_inv_nodes_large, h_general_sheaf)
            # alpha_norm = torch.bmm(alpha_norm, d_sqrt_inv_edges_large)
            # h_general_sheaf = alpha_norm.clamp(min=-1, max=1)

        #!!! Is this the correct reshape??? Please check!!
        h_general_sheaf_attributes = h_general_sheaf.reshape(-1)

        #create the big matrix from the dxd blocks  
        return h_general_sheaf_index, h_general_sheaf_attributes

    def init_hyperedge_attr(self, type, num_edges=None, x=None, hyperedge_index=None):
        #initialize hyperedge attributes either random or as the average of the nodes
        if type == 'rand':
            hyperedge_attr = torch.randn((num_edges, self.num_features)).to(self.device)
        elif type == 'avg':
            hyperedge_attr = scatter_mean(x[hyperedge_index[0]],hyperedge_index[1], dim=0)
        else:
            hyperedge_attr = None
        return hyperedge_attr

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        num_nodes = data.edge_index[0].max().item() + 1
        num_edges = data.edge_index[1].max().item() + 1
        #Iulia Qs: I don't think it's ok to generate random at each iteration
        # hyperedge_attr = torch.randn((num_edges, self.num_features)).to(self.device)
        
        #if we are at the first epoch, initialise the attribute, otherwise use the previous ones
        if self.hyperedge_attr is None:
            self.hyperedge_attr = self.init_hyperedge_attr(self.init_hedge, num_edges=num_edges, x=x, hyperedge_index=edge_index)

        #infer the sheaf as a sparse incidence matrix Nd x Ed
        # h_general_sheaf_index, h_general_sheaf_attributes = self.build_general_sheaf_incidence(x, self.hyperedge_attr, edge_index)


        x = self.lin(x) #N x num_features -> N x (d*num_features)
        hyperedge_attr = self.lin(self.hyperedge_attr)
        x = x.view((x.shape[0]*self.d, self.num_features)) # N x (d*num_features) -> (N * d) x num_features 
        hyperedge_attr = hyperedge_attr.view((hyperedge_attr.shape[0]*self.d, self.num_features))
        
        for i, conv in enumerate(self.convs[:-1]):
            if i == 0 or self.dynamic_sheaf:
                h_general_sheaf_index, h_general_sheaf_attributes = self.build_general_sheaf_incidence(x, hyperedge_attr, edge_index, self.general_sheaf_lin[i])
            x = F.elu(conv(x, hyperedge_index=h_general_sheaf_index, alpha=h_general_sheaf_attributes, num_nodes=num_nodes, num_edges=num_edges))
            x = F.dropout(x, p=self.dropout, training=self.training)
#         x = F.dropout(x, p=self.dropout, training=self.training)

        if self.dynamic_sheaf:
            h_general_sheaf_index, h_general_sheaf_attributes = self.build_general_sheaf_incidence(x, hyperedge_attr, edge_index, self.general_sheaf_lin[-1])
        x = self.convs[-1](x,  hyperedge_index=h_general_sheaf_index, alpha=h_general_sheaf_attributes, num_nodes=num_nodes, num_edges=num_edges)
        x = x.view(num_nodes, -1) # Nd x out_channels -> N x (d*out_channels)
        x = self.lin2(x) # N x (d*out_channels) -> N x num_channels

        return x


class HNHN(nn.Module):
    """
    """

    def __init__(self, args):
        super(HNHN, self).__init__()

        self.num_layers = args.All_num_layers
        self.dropout = args.dropout
        
        self.convs = nn.ModuleList()
        # two cases
        if self.num_layers == 1:
            self.convs.append(HNHNConv(args.num_features, args.MLP_hidden, args.num_classes,
                                       nonlinear_inbetween=args.HNHN_nonlinear_inbetween))
        else:
            self.convs.append(HNHNConv(args.num_features, args.MLP_hidden, args.MLP_hidden,
                                       nonlinear_inbetween=args.HNHN_nonlinear_inbetween))
            for _ in range(self.num_layers - 2):
                self.convs.append(HNHNConv(args.MLP_hidden, args.MLP_hidden, args.MLP_hidden,
                                           nonlinear_inbetween=args.HNHN_nonlinear_inbetween))
            self.convs.append(HNHNConv(args.MLP_hidden, args.MLP_hidden, args.num_classes,
                                       nonlinear_inbetween=args.HNHN_nonlinear_inbetween))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):

        x = data.x
        
        if self.num_layers == 1:
            conv = self.convs[0]
            x = conv(x, data)
            # x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            for i, conv in enumerate(self.convs[:-1]):
                x = F.relu(conv(x, data))
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, data)

        return x


class HCHA(nn.Module):
    """
    This model is proposed by "Hypergraph Convolution and Hypergraph Attention" (in short HCHA) and its convolutional layer 
    is implemented in pyg.


    self, in_channels, out_channels, use_attention=False, heads=1,
                 concat=True, negative_slope=0.2, dropout=0, bias=True,
                 **kwargs
    """

    def __init__(self, args):
        super(HCHA, self).__init__()

        self.num_layers = args.All_num_layers
        self.dropout = args.dropout  # Note that default is 0.6
        self.symdegnorm = args.HCHA_symdegnorm
        self.heads = args.heads
        self.num_features = args.num_features
        self.MLP_hidden = args.MLP_hidden // self.heads
        self.init_hedge = args.init_hedge
        self.hyperedge_attr = None

#         Note that add dropout to attention is default in the original paper
        self.convs = nn.ModuleList()
        #iulia Gs: should change here heads=args.heads?
        self.convs.append(HypergraphConv(args.num_features,
                                         self.MLP_hidden, use_attention=args.use_attention, heads = self.heads))
        
        #iulia Qs: add back the multi-layers?
        for _ in range(self.num_layers-2):
           self.convs.append(HypergraphConv(
               self.heads*self.MLP_hidden, self.MLP_hidden, use_attention=args.use_attention, heads = self.heads))
        # Output heads is set to 1 as default
        self.convs.append(HypergraphConv(
            self.heads*self.MLP_hidden, args.num_classes, use_attention=False))
        if args.cuda in [0, 1]:
            self.device = torch.device('cuda:'+str(args.cuda)
                              if torch.cuda.is_available() else 'cpu')

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def init_hyperedge_attr(self, type, num_edges=None, x=None, hyperedge_index=None):
        if type == 'rand':
            hyperedge_attr = torch.randn((num_edges, self.num_features)).to(self.device)
        elif type == 'avg':
            hyperedge_attr = scatter_mean(x[hyperedge_index[0]],hyperedge_index[1], dim=0)
        else:
            hyperedge_attr = None
        return hyperedge_attr

    def forward(self, data):

        x = data.x
        edge_index = data.edge_index
        num_nodes = data.edge_index[0].max().item() + 1

        num_edges = data.edge_index[1].max().item() + 1
        # print(num_nodes, num_edges)
        
        # hyperedge_attr = torch.randn((num_edges, self.num_features)).to(self.device)
        if self.hyperedge_attr is None:
            self.hyperedge_attr = self.init_hyperedge_attr(type=self.init_hedge, num_edges=num_edges, x=x, hyperedge_index=edge_index)
        # print(hyperedge_attr.shape)

        for i, conv in enumerate(self.convs[:-1]):
            # print(i)
            x = F.elu(conv(x, edge_index, hyperedge_attr = self.hyperedge_attr))
            x = F.dropout(x, p=self.dropout, training=self.training)
#         x = F.dropout(x, p=self.dropout, training=self.training)

        # print("Ok")
        x = self.convs[-1](x, edge_index)

        return x


class SetGNN(nn.Module):
    def __init__(self, args, norm=None):
        super(SetGNN, self).__init__()
        """
        args should contain the following:
        V_in_dim, V_enc_hid_dim, V_dec_hid_dim, V_out_dim, V_enc_num_layers, V_dec_num_layers
        E_in_dim, E_enc_hid_dim, E_dec_hid_dim, E_out_dim, E_enc_num_layers, E_dec_num_layers
        All_num_layers,dropout
        !!! V_in_dim should be the dimension of node features
        !!! E_out_dim should be the number of classes (for classification)
        """

#         Now set all dropout the same, but can be different
        self.All_num_layers = args.All_num_layers
        self.dropout = args.dropout
        self.aggr = args.aggregate
        self.NormLayer = args.normalization
        self.InputNorm = args.deepset_input_norm
        self.GPR = args.GPR
        self.LearnMask = args.LearnMask
#         Now define V2EConvs[i], V2EConvs[i] for ith layers
#         Currently we assume there's no hyperedge features, which means V_out_dim = E_in_dim
#         If there's hyperedge features, concat with Vpart decoder output features [V_feat||E_feat]
        self.V2EConvs = nn.ModuleList()
        self.E2VConvs = nn.ModuleList()
        self.bnV2Es = nn.ModuleList()
        self.bnE2Vs = nn.ModuleList()

        if self.LearnMask:
            self.Importance = Parameter(torch.ones(norm.size()))

        if self.All_num_layers == 0:
            self.classifier = MLP(in_channels=args.num_features,
                                  hidden_channels=args.Classifier_hidden,
                                  out_channels=args.num_classes,
                                  num_layers=args.Classifier_num_layers,
                                  dropout=self.dropout,
                                  Normalization=self.NormLayer,
                                  InputNorm=False)
        else:
            self.V2EConvs.append(HalfNLHconv(in_dim=args.num_features,
                                             hid_dim=args.MLP_hidden,
                                             out_dim=args.MLP_hidden,
                                             num_layers=args.MLP_num_layers,
                                             dropout=self.dropout,
                                             Normalization=self.NormLayer,
                                             InputNorm=self.InputNorm,
                                             heads=args.heads,
                                             attention=args.PMA))
            self.bnV2Es.append(nn.BatchNorm1d(args.MLP_hidden))
            self.E2VConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
                                             hid_dim=args.MLP_hidden,
                                             out_dim=args.MLP_hidden,
                                             num_layers=args.MLP_num_layers,
                                             dropout=self.dropout,
                                             Normalization=self.NormLayer,
                                             InputNorm=self.InputNorm,
                                             heads=args.heads,
                                             attention=args.PMA))
            self.bnE2Vs.append(nn.BatchNorm1d(args.MLP_hidden))
            for _ in range(self.All_num_layers-1):
                self.V2EConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
                                                 hid_dim=args.MLP_hidden,
                                                 out_dim=args.MLP_hidden,
                                                 num_layers=args.MLP_num_layers,
                                                 dropout=self.dropout,
                                                 Normalization=self.NormLayer,
                                                 InputNorm=self.InputNorm,
                                                 heads=args.heads,
                                                 attention=args.PMA))
                self.bnV2Es.append(nn.BatchNorm1d(args.MLP_hidden))
                self.E2VConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
                                                 hid_dim=args.MLP_hidden,
                                                 out_dim=args.MLP_hidden,
                                                 num_layers=args.MLP_num_layers,
                                                 dropout=self.dropout,
                                                 Normalization=self.NormLayer,
                                                 InputNorm=self.InputNorm,
                                                 heads=args.heads,
                                                 attention=args.PMA))
                self.bnE2Vs.append(nn.BatchNorm1d(args.MLP_hidden))
            if self.GPR:
                self.MLP = MLP(in_channels=args.num_features,
                               hidden_channels=args.MLP_hidden,
                               out_channels=args.MLP_hidden,
                               num_layers=args.MLP_num_layers,
                               dropout=self.dropout,
                               Normalization=self.NormLayer,
                               InputNorm=False)
                self.GPRweights = Linear(self.All_num_layers+1, 1, bias=False)
                self.classifier = MLP(in_channels=args.MLP_hidden,
                                      hidden_channels=args.Classifier_hidden,
                                      out_channels=args.num_classes,
                                      num_layers=args.Classifier_num_layers,
                                      dropout=self.dropout,
                                      Normalization=self.NormLayer,
                                      InputNorm=False)
            else:
                self.classifier = MLP(in_channels=args.MLP_hidden,
                                      hidden_channels=args.Classifier_hidden,
                                      out_channels=args.num_classes,
                                      num_layers=args.Classifier_num_layers,
                                      dropout=self.dropout,
                                      Normalization=self.NormLayer,
                                      InputNorm=False)


#         Now we simply use V_enc_hid=V_dec_hid=E_enc_hid=E_dec_hid
#         However, in general this can be arbitrary.


    def reset_parameters(self):
        for layer in self.V2EConvs:
            layer.reset_parameters()
        for layer in self.E2VConvs:
            layer.reset_parameters()
        for layer in self.bnV2Es:
            layer.reset_parameters()
        for layer in self.bnE2Vs:
            layer.reset_parameters()
        self.classifier.reset_parameters()
        if self.GPR:
            self.MLP.reset_parameters()
            self.GPRweights.reset_parameters()
        if self.LearnMask:
            nn.init.ones_(self.Importance)

    def forward(self, data):
        """
        The data should contain the follows
        data.x: node features
        data.edge_index: edge list (of size (2,|E|)) where data.edge_index[0] contains nodes and data.edge_index[1] contains hyperedges
        !!! Note that self loop should be assigned to a new (hyper)edge id!!!
        !!! Also note that the (hyper)edge id should start at 0 (akin to node id)
        data.norm: The weight for edges in bipartite graphs, correspond to data.edge_index
        !!! Note that we output final node representation. Loss should be defined outside.
        """
#             The data should contain the follows
#             data.x: node features
#             data.V2Eedge_index:  edge list (of size (2,|E|)) where
#             data.V2Eedge_index[0] contains nodes and data.V2Eedge_index[1] contains hyperedges

        x, edge_index, norm = data.x, data.edge_index, data.norm
        if self.LearnMask:
            norm = self.Importance*norm
        cidx = edge_index[1].min()
        edge_index[1] -= cidx  # make sure we do not waste memory
        reversed_edge_index = torch.stack(
            [edge_index[1], edge_index[0]], dim=0)
        if self.GPR:
            xs = []
            xs.append(F.relu(self.MLP(x)))
            for i, _ in enumerate(self.V2EConvs):
                x = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr))
#                 x = self.bnV2Es[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr)
                x = F.relu(x)
                xs.append(x)
#                 x = self.bnE2Vs[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = torch.stack(xs, dim=-1)
            x = self.GPRweights(x).squeeze()
            x = self.classifier(x)
        else:
            x = F.dropout(x, p=0.2, training=self.training) # Input dropout
            for i, _ in enumerate(self.V2EConvs):
                x = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr))
#                 x = self.bnV2Es[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(self.E2VConvs[i](
                    x, reversed_edge_index, norm, self.aggr))
#                 x = self.bnE2Vs[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.classifier(x)

        return x


class MLP_model(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    def __init__(self, args, InputNorm=False):
        super(MLP_model, self).__init__()
        in_channels = args.num_features
        hidden_channels = args.MLP_hidden
        out_channels = args.num_classes
        num_layers = args.All_num_layers
        dropout = args.dropout
        Normalization = args.normalization

        self.lins = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.InputNorm = InputNorm

        assert Normalization in ['bn', 'ln', 'None']
        if Normalization == 'bn':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        elif Normalization == 'ln':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.LayerNorm(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.LayerNorm(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        else:
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.Identity())
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for normalization in self.normalizations:
            if not (normalization.__class__.__name__ is 'Identity'):
                normalization.reset_parameters()

    def forward(self, data):
        x = data.x
        x = self.normalizations[0](x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i+1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


"""
The code below is directly adapt from the official implementation of UniGNN.
"""
# NOTE: can not tell which implementation is better statistically 

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def normalize_l2(X):
    """Row-normalize  matrix"""
    rownorm = X.detach().norm(dim=1, keepdim=True)
    scale = rownorm.pow(-1)
    scale[torch.isinf(scale)] = 0.
    X = X * scale
    return X



# v1: X -> XW -> AXW -> norm
class UniSAGEConv(nn.Module):

    def __init__(self, args, in_channels, out_channels, heads=8, dropout=0., negative_slope=0.2):
        super().__init__()
        # TODO: bias?
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.args = args

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

    def forward(self, X, vertex, edges):
        N = X.shape[0]
        
        # X0 = X # NOTE: reserved for skip connection

        X = self.W(X)

        Xve = X[vertex] # [nnz, C]
        Xe = scatter(Xve, edges, dim=0, reduce=self.args.first_aggregate) # [E, C]

        Xev = Xe[edges] # [nnz, C]
        Xv = scatter(Xev, vertex, dim=0, reduce=self.args.second_aggregate, dim_size=N) # [N, C]
        X = X + Xv 

        if self.args.use_norm:
            X = normalize_l2(X)

        # NOTE: concat heads or mean heads?
        # NOTE: normalize here?
        # NOTE: skip concat here?

        return X



# v1: X -> XW -> AXW -> norm
class UniGINConv(nn.Module):

    def __init__(self, args, in_channels, out_channels, heads=8, dropout=0., negative_slope=0.2):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.eps = nn.Parameter(torch.Tensor([0.]))
        self.args = args 
        
    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


    def forward(self, X, vertex, edges):
        N = X.shape[0]
        # X0 = X # NOTE: reserved for skip connection
        
        # v1: X -> XW -> AXW -> norm
        X = self.W(X) 

        Xve = X[vertex] # [nnz, C]
        Xe = scatter(Xve, edges, dim=0, reduce=self.args.first_aggregate) # [E, C]
        
        Xev = Xe[edges] # [nnz, C]
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N) # [N, C]
        X = (1 + self.eps) * X + Xv 

        if self.args.use_norm:
            X = normalize_l2(X)


        
        # NOTE: concat heads or mean heads?
        # NOTE: normalize here?
        # NOTE: skip concat here?

        return X



# v1: X -> XW -> AXW -> norm
class UniGCNConv(nn.Module):

    def __init__(self, args, in_channels, out_channels, heads=8, dropout=0., negative_slope=0.2):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)        
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.args = args 
        
    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

    def forward(self, X, vertex, edges):
        N = X.shape[0]
        degE = self.args.degE
        degV = self.args.degV
        
        # v1: X -> XW -> AXW -> norm
        
        X = self.W(X)

        Xve = X[vertex] # [nnz, C]
        Xe = scatter(Xve, edges, dim=0, reduce=self.args.first_aggregate) # [E, C]
        
        Xe = Xe * degE 

        Xev = Xe[edges] # [nnz, C]
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N) # [N, C]
        
        Xv = Xv * degV

        X = Xv 
        
        if self.args.use_norm:
            X = normalize_l2(X)

        # NOTE: skip concat here?

        return X



# v2: X -> AX -> norm -> AXW 
class UniGCNConv2(nn.Module):

    def __init__(self, args, in_channels, out_channels, heads=8, dropout=0., negative_slope=0.2):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=True)        
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.args = args 
        
    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

    def forward(self, X, vertex, edges):
        N = X.shape[0]
        degE = self.args.degE
        degV = self.args.degV

        # v3: X -> AX -> norm -> AXW 

        Xve = X[vertex] # [nnz, C]
        Xe = scatter(Xve, edges, dim=0, reduce=self.args.first_aggregate) # [E, C]
        
        Xe = Xe * degE 

        Xev = Xe[edges] # [nnz, C]
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N) # [N, C]
        
        Xv = Xv * degV

        X = Xv 

        if self.args.use_norm:
            X = normalize_l2(X)


        X = self.W(X)


        # NOTE: result might be slighly unstable
        # NOTE: skip concat here?

        return X



class UniGATConv(nn.Module):

    def __init__(self, args, in_channels, out_channels, heads=8, dropout=0., negative_slope=0.2, skip_sum=False):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        self.att_v = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_e = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attn_drop  = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.skip_sum = skip_sum
        self.args = args
        self.reset_parameters()

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

    def reset_parameters(self):
        glorot(self.att_v)
        glorot(self.att_e)

    def forward(self, X, vertex, edges):
        H, C, N = self.heads, self.out_channels, X.shape[0]
        
        # X0 = X # NOTE: reserved for skip connection

        X0 = self.W(X)
        X = X0.view(N, H, C)

        Xve = X[vertex] # [nnz, H, C]
        Xe = scatter(Xve, edges, dim=0, reduce=self.args.first_aggregate) # [E, H, C]


        alpha_e = (Xe * self.att_e).sum(-1) # [E, H, 1]
        a_ev = alpha_e[edges]
        alpha = a_ev # Recommed to use this
        alpha = self.leaky_relu(alpha)
        alpha = softmax(alpha, vertex, num_nodes=N)
        alpha = self.attn_drop( alpha )
        alpha = alpha.unsqueeze(-1)


        Xev = Xe[edges] # [nnz, H, C]
        Xev = Xev * alpha 
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N) # [N, H, C]
        X = Xv 
        X = X.view(N, H * C)

        if self.args.use_norm:
            X = normalize_l2(X)

        if self.skip_sum:
            X = X + X0 

        # NOTE: concat heads or mean heads?
        # NOTE: skip concat here?

        return X




__all_convs__ = {
    'UniGAT': UniGATConv,
    'UniGCN': UniGCNConv,
    'UniGCN2': UniGCNConv2,
    'UniGIN': UniGINConv,
    'UniSAGE': UniSAGEConv,
}



class UniGNN(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, nlayer, nhead, V, E):
        """UniGNN

        Args:
            args   (NamedTuple): global args
            nfeat  (int): dimension of features
            nhid   (int): dimension of hidden features, note that actually it\'s #nhid x #nhead
            nclass (int): number of classes
            nlayer (int): number of hidden layers
            nhead  (int): number of conv heads
            V (torch.long): V is the row index for the sparse incident matrix H, |V| x |E|
            E (torch.long): E is the col index for the sparse incident matrix H, |V| x |E|
        """
        super().__init__()
        Conv = __all_convs__[args.model_name]
        self.conv_out = Conv(args, nhid * nhead, nclass, heads=1, dropout=args.attn_drop)
        self.convs = nn.ModuleList(
            [ Conv(args, nfeat, nhid, heads=nhead, dropout=args.attn_drop)] +
            [Conv(args, nhid * nhead, nhid, heads=nhead, dropout=args.attn_drop) for _ in range(nlayer-2)]
        )
        self.V = V 
        self.E = E 
        act = {'relu': nn.ReLU(), 'prelu':nn.PReLU() }
        self.act = act[args.activation]
        self.input_drop = nn.Dropout(args.input_drop)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, X):
        V, E = self.V, self.E 
        
        X = self.input_drop(X)
        for conv in self.convs:
            X = conv(X, V, E)
            X = self.act(X)
            X = self.dropout(X)

        X = self.conv_out(X, V, E)      
        return F.log_softmax(X, dim=1)



class UniGCNIIConv(nn.Module):
    def __init__(self, args, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.args = args

    def reset_parameters(self):
        self.W.reset_parameters()
        
    def forward(self, X, vertex, edges, alpha, beta, X0):
        N = X.shape[0]
        degE = self.args.UniGNN_degE
        degV = self.args.UniGNN_degV

        Xve = X[vertex] # [nnz, C]
        Xe = scatter(Xve, edges, dim=0, reduce='mean') # [E, C], reduce is 'mean' here as default
        
        Xe = Xe * degE 

        Xev = Xe[edges] # [nnz, C]
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N) # [N, C]
        
        Xv = Xv * degV
        
        X = Xv 

        if self.args.UniGNN_use_norm:
            X = normalize_l2(X)

        Xi = (1-alpha) * X + alpha * X0
        X = (1-beta) * Xi + beta * self.W(Xi)


        return X



class UniGCNII(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, nlayer, nhead, V, E):
        """UniGNNII

        Args:
            args   (NamedTuple): global args
            nfeat  (int): dimension of features
            nhid   (int): dimension of hidden features, note that actually it\'s #nhid x #nhead
            nclass (int): number of classes
            nlayer (int): number of hidden layers
            nhead  (int): number of conv heads
            V (torch.long): V is the row index for the sparse incident matrix H, |V| x |E|
            E (torch.long): E is the col index for the sparse incident matrix H, |V| x |E|
        """
        super().__init__()
        self.V = V 
        self.E = E 
        nhid = nhid * nhead
        act = {'relu': nn.ReLU(), 'prelu':nn.PReLU() }
        self.act = act['relu'] # Default relu
        self.input_drop = nn.Dropout(0.6) # 0.6 is chosen as default
        self.dropout = nn.Dropout(0.2) # 0.2 is chosen for GCNII

        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(nfeat, nhid))
        for _ in range(nlayer):
            self.convs.append(UniGCNIIConv(args, nhid, nhid))
        self.convs.append(torch.nn.Linear(nhid, nclass))
        self.reg_params = list(self.convs[1:-1].parameters())
        self.non_reg_params = list(self.convs[0:1].parameters())+list(self.convs[-1:].parameters())
        self.dropout = nn.Dropout(0.2) # 0.2 is chosen for GCNII
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        
    def forward(self, data):
        x = data.x
        V, E = self.V, self.E 
        lamda, alpha = 0.5, 0.1 
        x = self.dropout(x)
        x = F.relu(self.convs[0](x))
        x0 = x 
        for i,con in enumerate(self.convs[1:-1]):
            x = self.dropout(x)
            beta = math.log(lamda/(i+1)+1)
            x = F.relu(con(x, V, E, alpha, beta, x0))
        x = self.dropout(x)
        x = self.convs[-1](x)
        return x