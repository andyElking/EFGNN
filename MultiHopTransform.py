import sys

import torch
import torch_geometric.transforms as T
from torch_geometric.utils import coalesce, remove_self_loops, to_edge_index, to_torch_csr_tensor, to_dense_adj, degree, \
    add_self_loops
from torch import sparse
from torch.nn.functional import pad


class OneTwoHopCSR(object):
    def __call__(self, data, *args, **kwargs):
        """
        Augments the data with CSR adjacencies and precomputes several other things which are used often by EFGNN
        Args:
            data:
            *args:
            **kwargs:

        Returns:

        """
        edge_idx = data.edge_index
        n = data.num_nodes

        edge_idx, _ = remove_self_loops(edge_idx)
        data.edge_index = edge_idx

        one_hop_csr = to_torch_csr_tensor(edge_idx)

        renorm_edge_idx, _ = add_self_loops(edge_idx, num_nodes=n)
        renorm_csr = to_torch_csr_tensor(renorm_edge_idx, size=n)

        two_hop_csr = one_hop_csr @ one_hop_csr

        edge_idx2, _ = to_edge_index(two_hop_csr)
        edge_idx2, _ = remove_self_loops(edge_idx2)
        edge_idx2 = coalesce(edge_idx2, num_nodes=n)
        two_hop_csr = to_torch_csr_tensor(edge_idx2, size=n)

        edge_idx12 = torch.cat([edge_idx, edge_idx2], dim=1)

        values_1hop = torch.tensor([[1, 0]], dtype=torch.float).expand(edge_idx.shape[1], 2)
        values_2hop = torch.tensor([[0, 1]], dtype=torch.float).expand(edge_idx2.shape[1], 2)
        hop_one_hot = torch.cat([values_1hop, values_2hop], dim=0)

        # edge_idx12, hop_one_hot = coalesce(edge_idx12, hop_one_hot)
        hop12_coo = torch.sparse_coo_tensor(indices=edge_idx12, values=hop_one_hot).coalesce()
        edge_idx12 = hop12_coo.indices()

        deg1 = degree(edge_idx[0], num_nodes=n).unsqueeze(1)
        deg2 = degree(edge_idx2[0], num_nodes=n).unsqueeze(1)
        deg12 = degree(edge_idx12[0], num_nodes=n).unsqueeze(1)
        renorm_deg = 1 + deg1

        data.deg1 = deg1
        data.h1_deg_pows = {
            -2: torch.nan_to_num(torch.pow(deg1, exponent=-2), nan=0, posinf=0, neginf=0),
            -1.5: torch.nan_to_num(torch.pow(deg1, exponent=-1.5), nan=0, posinf=0, neginf=0),
            -1: torch.nan_to_num(torch.pow(deg1, exponent=-1), nan=0, posinf=0, neginf=0),
            -0.5: torch.nan_to_num(torch.pow(deg1, exponent=-0.5), nan=0, posinf=0, neginf=0),
            0: 1,
            0.5: torch.nan_to_num(torch.pow(deg1, exponent=0.5), nan=0, posinf=0, neginf=0),
            1: deg1,
            2: torch.nan_to_num(torch.pow(deg1, exponent=2), nan=0, posinf=0, neginf=0),
        }
        data.deg2 = deg2
        data.h2_deg_pows = {
            -2: torch.nan_to_num(torch.pow(deg2, exponent=-2), nan=0, posinf=0, neginf=0),
            -1.5: torch.nan_to_num(torch.pow(deg2, exponent=-1.5), nan=0, posinf=0, neginf=0),
            -1: torch.nan_to_num(torch.pow(deg2, exponent=-1), nan=0, posinf=0, neginf=0),
            -0.5: torch.nan_to_num(torch.pow(deg2, exponent=-0.5), nan=0, posinf=0, neginf=0),
            0: 1,
            0.5: torch.nan_to_num(torch.pow(deg2, exponent=0.5), nan=0, posinf=0, neginf=0),
            1: deg2,
            2: torch.nan_to_num(torch.pow(deg2, exponent=2), nan=0, posinf=0, neginf=0),
        }

        data.renorm_deg = renorm_deg
        data.renorm_deg_pows = {
            -2: torch.nan_to_num(torch.pow(renorm_deg, exponent=-2), nan=0, posinf=0, neginf=0),
            -1.5: torch.nan_to_num(torch.pow(renorm_deg, exponent=-1.5), nan=0, posinf=0, neginf=0),
            -1: torch.nan_to_num(torch.pow(renorm_deg, exponent=-1), nan=0, posinf=0, neginf=0),
            -0.5: torch.nan_to_num(torch.pow(renorm_deg, exponent=-0.5), nan=0, posinf=0, neginf=0),
            0: 1,
            0.5: torch.nan_to_num(torch.pow(renorm_deg, exponent=0.5), nan=0, posinf=0, neginf=0),
            1: renorm_deg,
            2: torch.nan_to_num(torch.pow(renorm_deg, exponent=2), nan=0, posinf=0, neginf=0),
        }

        data.deg12 = deg12
        data.h12_deg_pows = {
            -2: torch.nan_to_num(torch.pow(deg12, exponent=-2), nan=0, posinf=0, neginf=0),
            -1.5: torch.nan_to_num(torch.pow(deg12, exponent=-1.5), nan=0, posinf=0, neginf=0),
            -1: torch.nan_to_num(torch.pow(deg12, exponent=-1), nan=0, posinf=0, neginf=0),
            -0.5: torch.nan_to_num(torch.pow(deg12, exponent=-0.5), nan=0, posinf=0, neginf=0),
            0: 1,
            0.5: torch.nan_to_num(torch.pow(deg12, exponent=0.5), nan=0, posinf=0, neginf=0),
            1: deg12,
            2: torch.nan_to_num(torch.pow(deg12, exponent=2), nan=0, posinf=0, neginf=0),
        }

        data.edge_index2 = edge_idx2
        data.hop12_coo = hop12_coo
        data.one_hop_csr = one_hop_csr
        data.renorm_csr = renorm_csr
        data.two_hop_csr = two_hop_csr

        return data
