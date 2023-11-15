import torch
import torch.nn as nn
from torch_geometric.utils import remove_self_loops, to_torch_csr_tensor, to_dense_adj, degree, add_self_loops
import torch.nn.functional as F
from math import sqrt


class AttLayer(nn.Module):
    def __init__(self, input_dim, emb_dim, use_cos_sim, use_dist, simple_layer2=False, two_hops=False):
        super(AttLayer, self).__init__()
        self.num_embs = 2 + int(use_cos_sim) + int(use_dist)
        self.emb_dim = emb_dim
        self.two_hops = two_hops

        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, self.num_embs * emb_dim, bias=False),
            nn.LayerNorm(self.num_embs * emb_dim, elementwise_affine=True),
            nn.LeakyReLU(0.02)
        )

        self.cos_sim_layer = nn.Linear(emb_dim, emb_dim) if use_cos_sim else None

        self.dist_layer = nn.Linear(emb_dim, emb_dim) if use_dist else None

        if simple_layer2:
            self.layer2 = nn.Sequential(
                nn.Linear(int(use_cos_sim) + 2*int(two_hops) + (2 + int(use_dist)) * emb_dim, 1),
                nn.Tanh()
            )
        else:
            self.layer2 = nn.Sequential(
                nn.Linear(int(use_cos_sim) + 2*int(two_hops) + (2 + int(use_dist)) * emb_dim, emb_dim),
                nn.LeakyReLU(0.1),
                nn.Linear(emb_dim, 1),
                nn.Tanh()
            )

    def forward(self, data):

        edge_idx = data.hop12_coo.indices() if self.two_hops else data.edge_index
        degs = data.h12_deg_pows if self.two_hops else data.h1_deg_pows
        num_edges = edge_idx.shape[1]

        embs = self.layer1(data.x)
        embs = list(torch.split(embs, self.emb_dim, dim=1))

        att_concats = []

        if self.cos_sim_layer is not None:
            cos_sim_emb = self.cos_sim_layer(embs[0])
            cos_sim_emb = F.normalize(cos_sim_emb, p=2, dim=1)
            embs = embs[1:]
            cos_sim = cos_sim_emb[edge_idx[0]] * cos_sim_emb[edge_idx[1]]
            cos_sim = 10 * torch.sum(cos_sim, dim=1, keepdim=True)
            assert cos_sim.shape == (num_edges, 1), f"Wrong cos_sim shape: {cos_sim.shape}, num_edges = {num_edges}"
            att_concats.append(cos_sim)

        if self.dist_layer is not None:
            dist_embs = self.dist_layer(embs[0])
            embs = embs[1:]
            dist = dist_embs[edge_idx[0]] - dist_embs[edge_idx[1]]
            dist = torch.abs(dist)
            assert dist.shape == (num_edges, self.emb_dim)
            att_concats.append(dist)

        if self.two_hops:
            edge_attr = data.hop12_coo.values()
            att_concats.append(edge_attr)

        assert len(embs) == 2
        for i, emb in enumerate(embs):
            att_concats.append(emb[edge_idx[i]])

        att_concats = torch.cat(att_concats, dim=1)

        out = self.layer2(att_concats)
        assert out.shape == (num_edges, 1)

        return out, edge_idx, degs
