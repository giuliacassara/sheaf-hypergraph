import torch
# from layers import HypergraphDiagSheafConv, HypergraphOrthoSheafConv, HypergraphGeneralSheafConv
from models import *
from argparse import Namespace
from utils import print_a_colored_ndarray
import pdb
import numpy as np
from torch_geometric.data import Data

import torch
import numpy as np

from scipy import sparse as sp


def get_test_config():
    return {
        'All_num_layers': 3,
        'MLP_hidden': 64,
        'init_hedge': 'avg',
        'cuda': -1,
        'dropout': 0.0,
        'num_features': 16,
        'num_classes': 32,
        'heads': 3,
        'sheaf_normtype': 'block_norm',
        'sheaf_pred_block': 'transformer',
        'sheaf_transformer_head': 8,#MLP_hidden needs to divide this
        'dynamic_sheaf': False,
        'sheaf_left_proj': False,
        'sheaf_act': 'sigmoid',
        'sheaf_dropout': False,
        'sheaf_special_head': False


    }

def is_valid_permutation_matrix(P: np.ndarray, n: int):
    #from: https://github.com/twitter-research/neural-sheaf-diffusion/blob/master/lib/perm_utils.py
    valid = True
    valid &= P.ndim == 2
    valid &= P.shape[0] == n
    valid &= np.all(P.sum(0) == np.ones(n))
    valid &= np.all(P.sum(1) == np.ones(n))
    valid &= np.all(P.max(0) == np.ones(n))
    valid &= np.all(P.max(1) == np.ones(n))
    if n > 1:
        valid &= np.all(P.min(0) == np.zeros(n))
        valid &= np.all(P.min(1) == np.zeros(n))
        valid &= not np.array_equal(P, np.eye(n))
    return valid

def generate_permutation_matrices(size, amount=10):
    #from: https://github.com/twitter-research/neural-sheaf-diffusion/blob/master/lib/perm_utils.py
    Ps = list()
    random_state = np.random.RandomState()
    count = 0
    while count < amount:
        I = np.eye(size)
        perm = random_state.permutation(size)
        P = I[perm]
        if is_valid_permutation_matrix(P, size):
            Ps.append(P)
            count += 1

    return Ps

def permute_hypergraph(graph: Data, P: np.ndarray) -> Data:
    #modify based on
    #this code: https://github.com/twitter-research/neural-sheaf-diffusion/blob/master/lib/perm_utils.py
    assert graph.edge_attr is None

    # Check validity of permutation matrix
    n = graph.x.size(0)
    m = graph.edge_index[1].max()+1
    if not is_valid_permutation_matrix(P, n):
        raise AssertionError

    # Apply permutation to features
    x = graph.x.numpy()
    x_perm = torch.FloatTensor(P @ x)

    # Apply perm to labels, if per-node
    if graph.y is None:
        y_perm = None
    elif graph.y.size(0) == n:
        y = graph.y.numpy()
        y_perm = torch.tensor(P @ y)
    else:
        y_perm = graph.y.clone().detach()

    # Apply permutation to adjacencies, if any
    if graph.edge_index.size(1) > 0:
        inps = (np.ones(graph.edge_index.size(1)), (graph.edge_index[0].numpy(), graph.edge_index[1].numpy()))
        H = sp.csr_matrix(inps, shape=(n, m))
        P = sp.csr_matrix(P)
        H_perm = P.dot(H).tocoo()
        edge_index_perm = torch.LongTensor(np.vstack((H_perm.row, H_perm.col)))
    else:
        edge_index_perm = graph.edge_index.clone().detach()

    # Instantiate new graph
    graph_perm = Data(x=x_perm, edge_index=edge_index_perm, y=y_perm)

    return graph_perm


# Test diagonal:
def test_sheaf_conv_diag():
    torch.random.manual_seed(0)

    args = get_test_config()
    in_channels = args['num_features']
    out_channels = args['num_classes']
    d = args['heads']
    args = Namespace(**args)
    print(args)

    hyperedge_index = torch.tensor([[0, 0, 1, 1, 2, 3], [0, 1, 0, 1, 0, 1]])
    num_nodes = hyperedge_index[0].max().item() + 1
    num_edges = hyperedge_index[1].max().item() + 1
    x = torch.randn((num_nodes*d, in_channels))
    hyperedge_attr = torch.randn((num_edges, in_channels))

    diag_sheaf_model = DiagSheafs(args)
    # diag_sheaf_conv = HypergraphDiagSheafConv(in_channels, out_channels, d=3, device=device)


    h_sheaf_index, h_sheaf_attributes = diag_sheaf_model.build_sheaf_incidence(x, hyperedge_attr, hyperedge_index, layer_idx=0, debug=True)

    #create the sparse matrix denoting H
    h_sheaf_sparse = torch.sparse_coo_tensor(h_sheaf_index, h_sheaf_attributes)
    h_sheaf_dense = h_sheaf_sparse.to_dense()

    assert h_sheaf_dense.size() == (num_nodes*d, num_edges*d)

    print (diag_sheaf_model.h_sheaf)
    print(hyperedge_index)

    print_a_colored_ndarray(h_sheaf_dense.detach().numpy(), d=d)

    # out = diag_sheaf_conv(x, h_sheaf_index, alpha=h_sheaf_attributes, num_nodes=num_nodes*d, num_edges=num_edges*d)
    # assert out.size() == (num_nodes*d, out_channels)

def test_equivariance_sheaf_diag():
    torch.random.manual_seed(0)
    np.random.seed(0)

    args = get_test_config()
    in_channels = args['num_features']
    out_channels = args['num_classes']
    d = args['heads']
    args = Namespace(**args)
    print(args)


    hyperedge_index = torch.tensor([[0, 0, 1, 1, 2, 3], [0, 1, 0, 1, 0, 1]])
    num_nodes = hyperedge_index[0].max().item() + 1
    num_edges = hyperedge_index[1].max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    with torch.no_grad():
        diag_sheaf_model = DiagSheafs(args)
        # diag_sheaf_conv = HypergraphDiagSheafConv(in_channels, out_channels, d=3, device=device)
        data = Data(x=x, edge_index=hyperedge_index)

        P = generate_permutation_matrices(size=num_nodes, amount=1)[0]
        perm_data = permute_hypergraph(data, P)

        # this is just to test the sheaf prediction
        # h_sheaf_index, h_sheaf_attributes = diag_sheaf_model.build_sheaf_incidence(x, hyperedge_attr, hyperedge_index, layer_idx=0, debug=False)

        #this is to test the perm equiv of prediction, sheaf generation is inside anyway
        out = diag_sheaf_model(data)
        out = torch.FloatTensor(P.astype(np.float64) @ out.numpy().astype(np.float64))
        perm_out = diag_sheaf_model(perm_data)
        print(out.mean(), perm_out.mean())

        assert(torch.allclose(out, perm_out, atol=1e-6))
        print("model DiagSheafs is permutation equivariant")


# Test general:
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




def test_equivariance_sheaf_general():
    torch.random.manual_seed(0)
    np.random.seed(0)

    args = get_test_config()
    in_channels = args['num_features']
    out_channels = args['num_classes']
    d = args['heads']
    args = Namespace(**args)
    print(args)


    hyperedge_index = torch.tensor([[0, 0, 1, 1, 2, 3], [0, 1, 0, 1, 0, 1]])
    num_nodes = hyperedge_index[0].max().item() + 1
    num_edges = hyperedge_index[1].max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    with torch.no_grad():
        general_sheaf_model = GeneralSheafs(args)
        # diag_sheaf_conv = HypergraphDiagSheafConv(in_channels, out_channels, d=3, device=device)
        data = Data(x=x, edge_index=hyperedge_index)

        P = generate_permutation_matrices(size=num_nodes, amount=1)[0]
        perm_data = permute_hypergraph(data, P)

        # this is just to test the sheaf prediction
        # h_sheaf_index, h_sheaf_attributes = diag_sheaf_model.build_sheaf_incidence(x, hyperedge_attr, hyperedge_index, layer_idx=0, debug=False)

        #this is to test the perm equiv of prediction, sheaf generation is inside anyway
        out = general_sheaf_model(data)
        out = torch.FloatTensor(P.astype(np.float64) @ out.numpy().astype(np.float64))
        perm_out = general_sheaf_model(perm_data)


        print(out.mean(), perm_out.mean())
        print(torch.abs(out-perm_out).max())
        assert(torch.allclose(out, perm_out, atol=1e-6))
        print("model GeneralSheafs is permutation equivariant")



# Test ortho  
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


def test_equivariance_sheaf_ortho():
    torch.random.manual_seed(0)
    np.random.seed(0)

    args = get_test_config()
    in_channels = args['num_features']
    out_channels = args['num_classes']
    d = args['heads']
    args = Namespace(**args)
    print(args)


    hyperedge_index = torch.tensor([[0, 0, 1, 1, 2, 3], [0, 1, 0, 1, 0, 1]])
    num_nodes = hyperedge_index[0].max().item() + 1
    num_edges = hyperedge_index[1].max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    with torch.no_grad():
        ortho_sheaf_model = OrthoSheafs(args)
        # diag_sheaf_conv = HypergraphDiagSheafConv(in_channels, out_channels, d=3, device=device)
        data = Data(x=x, edge_index=hyperedge_index)

        P = generate_permutation_matrices(size=num_nodes, amount=1)[0]
        perm_data = permute_hypergraph(data, P)

        # this is just to test the sheaf prediction
        # h_sheaf_index, h_sheaf_attributes = diag_sheaf_model.build_sheaf_incidence(x, hyperedge_attr, hyperedge_index, layer_idx=0, debug=False)

        #this is to test the perm equiv of prediction, sheaf generation is inside anyway
        out = ortho_sheaf_model(data)
        out = torch.FloatTensor(P.astype(np.float64) @ out.numpy().astype(np.float64))
        perm_out = ortho_sheaf_model(perm_data)
        print(out.mean(), perm_out.mean())
        assert(torch.allclose(out, perm_out, atol=1e-6))
        print("model OrthoSheafs is permutation equivariant")

if __name__ == '__main__':
    # test_sheaf_conv_diag()
    # test_sheaf_conv_general()
    # test_sheaf_conv_ortho()
    # test_equivariance_sheaf_diag()
    test_equivariance_sheaf_general()
    # test_equivariance_sheaf_ortho()

