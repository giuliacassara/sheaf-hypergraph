import torch
# from layers import HypergraphDiagSheafConv, HypergraphOrthoSheafConv, HypergraphGeneralSheafConv
from models import *
from argparse import Namespace
from utils import print_a_colored_ndarray
import pdb
import numpy as np
from torch_geometric.data import Data



def get_test_config():
    return {
        'All_num_layers': 3,
        'MLP_hidden': 64,
        'init_hedge': 'rand',
        'cuda': -1,
        'dropout': 0.6

    }


def test_sheaf_conv_diag():
    torch.manual_seed(0)
    in_channels, out_channels = (16, 32)
    d = 3

    args = get_test_config()
    args['num_features'] = in_channels
    args['num_classes'] = out_channels
    args['heads'] = d 
    args = Namespace(**args)
    print(args)

    hyperedge_index = torch.tensor([[0, 0, 1, 1, 2, 3], [0, 1, 0, 1, 0, 1]])
    num_nodes = hyperedge_index[0].max().item() + 1
    num_edges = hyperedge_index[1].max().item() + 1
    x = torch.randn((num_nodes*d, in_channels))
    hyperedge_attr = torch.randn((num_edges, in_channels))

    diag_sheaf_model = DiagSheafs(args)
    # diag_sheaf_conv = HypergraphDiagSheafConv(in_channels, out_channels, d=3, device=device)


    h_sheaf_index, h_sheaf_attributes = diag_sheaf_model.build_sheaf_incidence(x, hyperedge_attr, hyperedge_index, debug=True)

    #create the sparse matrix denoting H
    h_sheaf_sparse = torch.sparse_coo_tensor(h_sheaf_index, h_sheaf_attributes)
    h_sheaf_dense = h_sheaf_sparse.to_dense()

    assert h_sheaf_dense.size() == (num_nodes*d, num_edges*d)

    print (diag_sheaf_model.h_sheaf)
    print(hyperedge_index)

    print_a_colored_ndarray(h_sheaf_dense.detach().numpy(), d=d)

    # out = diag_sheaf_conv(x, h_sheaf_index, alpha=h_sheaf_attributes, num_nodes=num_nodes*d, num_edges=num_edges*d)
    # assert out.size() == (num_nodes*d, out_channels)
    
def test_sheaf_conv_general():
    torch.manual_seed(0)
    in_channels, out_channels = (16, 32)
    d = 3

    args = get_test_config()
    args['num_features'] = in_channels
    args['num_classes'] = out_channels
    args['heads'] = d 
    args = Namespace(**args)
    print(args)

    hyperedge_index = torch.tensor([[0, 0, 1, 1, 2, 3], [0, 1, 0, 1, 0, 1]])
    num_nodes = hyperedge_index[0].max().item() + 1
    num_edges = hyperedge_index[1].max().item() + 1
    x = torch.randn((num_nodes, in_channels))
    hyperedge_attr = torch.randn((num_edges, in_channels))

    general_sheaf_model = GeneralSheafs(args)
    # diag_sheaf_conv = HypergraphDiagSheafConv(in_channels, out_channels, d=3, device=device)


    h_sheaf_index, h_sheaf_attributes = general_sheaf_model.build_general_sheaf_incidence(x, hyperedge_attr, hyperedge_index, debug=True)

    #create the sparse matrix denoting H
    h_sheaf_sparse = torch.sparse_coo_tensor(h_sheaf_index, h_sheaf_attributes)
    h_sheaf_dense = h_sheaf_sparse.to_dense()

    assert h_sheaf_dense.size() == (num_nodes*d, num_edges*d)

    #debug regarding H sheaf creation
    print("Learnt sheaf H:")
    print (general_sheaf_model.h_general_sheaf)
    print("Hedge indices:")
    print(hyperedge_index)

    print_a_colored_ndarray(h_sheaf_dense.detach().numpy(), d=d)

    #debug regarding inference
    data = Data(x=x, edge_index=hyperedge_index)
    out = general_sheaf_model(data)

    # out = diag_sheaf_conv(x, h_sheaf_index, alpha=h_sheaf_attributes, num_nodes=num_nodes*d, num_edges=num_edges*d)
    # assert out.size() == (num_nodes*d, out_channels)

    
def test_sheaf_conv_ortho():
    torch.manual_seed(0)
    in_channels, out_channels = (16, 32)
    d = 3

    args = get_test_config()
    args['num_features'] = in_channels
    args['num_classes'] = out_channels
    args['heads'] = d 
    args = Namespace(**args)
    print(args)

    hyperedge_index = torch.tensor([[0, 0, 1, 1, 2, 3], [0, 1, 0, 1, 0, 1]])
    num_nodes = hyperedge_index[0].max().item() + 1
    num_edges = hyperedge_index[1].max().item() + 1
    x = torch.randn((num_nodes, in_channels))
    hyperedge_attr = torch.randn((num_edges, in_channels))

    general_sheaf_model = OrthoSheafs(args)
    # diag_sheaf_conv = HypergraphDiagSheafConv(in_channels, out_channels, d=3, device=device)


    h_sheaf_index, h_sheaf_attributes = general_sheaf_model.build_ortho_sheaf_incidence(x, hyperedge_attr, hyperedge_index, debug=True)

    #create the sparse matrix denoting H
    h_sheaf_sparse = torch.sparse_coo_tensor(h_sheaf_index, h_sheaf_attributes)
    h_sheaf_dense = h_sheaf_sparse.to_dense()

    assert h_sheaf_dense.size() == (num_nodes*d, num_edges*d)

    #debug regarding H sheaf creation
    print("Learnt sheaf H:")
    print (general_sheaf_model.h_ortho_sheaf)
    print("Hedge indices:")
    print(hyperedge_index)

    print_a_colored_ndarray(h_sheaf_dense.detach().numpy(), d=d)

    #debug regarding inference
    data = Data(x=x, edge_index=hyperedge_index)
    out = general_sheaf_model(data)

    # out = diag_sheaf_conv(x, h_sheaf_index, alpha=h_sheaf_attributes, num_nodes=num_nodes*d, num_edges=num_edges*d)
    # assert out.size() == (num_nodes*d, out_channels)

if __name__ == '__main__':
    # test_sheaf_conv_diag()
    # test_sheaf_conv_general()
    test_sheaf_conv_ortho()

