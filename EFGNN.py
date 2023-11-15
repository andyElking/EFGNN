import torch
import torch.nn as nn
from torch_geometric.utils import remove_self_loops, to_torch_csr_tensor, to_dense_adj, degree, add_self_loops, scatter
import torch.nn.functional as F
from math import sqrt
from AttLayer import AttLayer

# These use the old system, which is slightly more confusing. They still work, but I recommend using the new filters
# described below.
old_flts = [{"I": 1},  # 1
            {"I": 1, "h1": (1, -1, 0)}, {"I": 1, "h1": (1, -0.5, -0.5)}, {"I": 1, "h1": (1, 0, -1)},  # 4
            {"I": 1, "h1": (-1, -1, 0)}, {"I": 1, "h1": (-1, -0.5, -0.5)}, {"I": 1, "h1": (-1, 0, -1)},  # 7
            {"h1": (1, -1, 0)}, {"h1": (1, -0.5, 0)}, {"h1": (1, 0, 0)}, {"h1": (1, 0.5, 0)}, {"h1": (1, 1, 0)},  # 12
            {"I": 1, "h2": (1, -1, 0)}, {"I": 1, "h2": (1, -0.5, -0.5)}, {"I": 1, "h2": (1, 0, -1)},  # 15
            {"I": 1, "h2": (-1, -1, 0)}, {"I": 1, "h2": (-1, -0.5, -0.5)}, {"I": 1, "h2": (-1, 0, -1)},  # 18
            {"h2": (1, -1, 0)}, {"h2": (1, -0.5, 0)}, {"h2": (1, 0, 0)}, {"h2": (1, 0.5, 0)}, {"h2": (1, 1, 0)}]  # 23


# A filter is represented by a list of 5-tuples, each tuple describing one term in the sum which makes up a filter.
# Tuples are of the form (l, k, r, p, q), where:
# l is the number of hops, k is the multiplier, r (bool) renormalisation trick, p&q determine the normalisation
# if l = 0, we get k*I (identity / ego nodes)
# if l = 1 we get D^p A D^q
# if l >= 2 we get D^p (A D^{-1})^{l-1} A D^q
# if r = True replace A by A+I and D by D+I (renormalisation trick)
# finally a filter is a sum of

class EFGNN(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, filters, prop2_filts, use_decoder=False, dp1=0.5, dp2=0.5,
                 use_alpha=True, use_deg=True, noise_mult=0.1, use_att=False, use_act=True):
        """
        See forward for an explanation of what everything does
        Args:
            input_dim: num input features
            hid_dim:
            output_dim: num classes
            filters: which filters to use, see above
            prop2_filts: which 2nd propagation filters to use (see report...)
            use_decoder: whether to use an additional 2-layer MLP at the end
            dp1: dropout after first layer
            dp2: dropout in the decoder (only relevant if use_decoder==True)
            use_alpha: whether to use alpha as a learnable param, otherwise it is fixed
            use_deg: whether to inject the degree as a feature
            noise_mult: inject Gaussian noise into the first layer embeddings (noise mult gives st dev)
            use_att: whether to use attention
            use_act: if True, use leakyReLU(0.01) else id
        """
        super(EFGNN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.filters = filters
        self.prop2_filts = prop2_filts
        self.num_in_flts = len(self.filters)
        self.num_out_flts = self.num_in_flts + len(self.prop2_filts)
        self.dp1 = dp1
        self.noise_mult = noise_mult

        self.layer1 = nn.Linear(input_dim, self.num_in_flts * hid_dim)

        self.leaky = nn.LeakyReLU(0.01) if use_act else None
        self.lay_2_dim = 2*hid_dim if use_decoder else output_dim

        self.out_layers = nn.ModuleList([nn.Linear(hid_dim, self.lay_2_dim) for i in range(self.num_out_flts)])

        def get2prop_layer():
            return nn.Linear(self.num_in_flts * hid_dim, len(self.prop2_filts) * hid_dim)

        self.double_prop_layer = get2prop_layer() if len(self.prop2_filts) > 0 else None

        self.decoder = None
        if use_decoder:
            self.decoder = nn.Sequential(
                nn.LeakyReLU(0.03),
                nn.Dropout(p=dp2),
                nn.Linear(2*hid_dim, output_dim)
            )

        self.att_layer = AttLayer(input_dim, hid_dim//4, use_cos_sim=True, use_dist=True) if use_att else None

        self.deg_layer = None

        self.alpha = torch.ones((self.num_out_flts + int(use_deg),), dtype=torch.float, device=self.device)
        self.gamma = torch.tensor(1.0, dtype=torch.float, device=self.device)
        self.alpha_params = []

        if use_deg:
            self.deg_layer = nn.Linear(1, self.lay_2_dim)
            self.alpha = torch.ones((self.num_out_flts + 1,), dtype=torch.float, device=self.device)

        if use_alpha:
            self.alpha = nn.Parameter(self.alpha, requires_grad=True)
            self.gamma = nn.Parameter(self.gamma, requires_grad=True)
            self.alpha_params = [self.alpha]

        self.layer_params = [self.gamma]
        for lay in [self.layer1, self.double_prop_layer, self.deg_layer, self.decoder, self.out_layers]:
            if lay is not None:
                self.layer_params.extend(lay.parameters())

    def apply_filter(self, filt, z, data, att=None):
        """
        Explained at the top of this file.
        :param data: the data object (which has all the csrs and powers of degree precomputed)
        :param filt: the filter (either in old form (dict) or in new form (list))
        :param z: node feature embedding
        :param att: attention-based adjacency (not very helpful)
        :return:
        """
        res = torch.zeros(z.shape, dtype=torch.float, device=self.device)

        if isinstance(filt, list):
            for l, k, r, p, q in filt:
                if l == 0:
                    res = torch.add(res, alpha=k, other=z)
                adj = data.renorm_csr if r else data.one_hop_csr
                degs = data.renorm_deg_pows if r else data.h1_deg_pows
                diffused = adj @ (degs[q] * z)
                for i in range(l-1):
                    diffused = adj @ diffused
                diffused = degs[p] * diffused
                res = torch.add(res, alpha=k, other=diffused)
            return res

        if "I" in filt:
            res = res + filt["I"] * z
        for hop in ["h1", "h2"]:
            if hop in filt:
                mult, j_exp, i_exp = filt[hop]
                adj = data.one_hop_csr
                degs = data.h1_deg_pows
                diffused = adj @ (degs[j_exp] * z)
                if hop == "h2":
                    diffused = adj @ (degs[-1] * diffused)
                diffused = degs[i_exp] * diffused
                res = torch.add(res, alpha=mult, other=diffused)
        if "h2!" in filt:
            mult, j_exp, i_exp = filt["h2!"]
            adj = data.two_hop_csr
            degs = data.h2_deg_pows
            diffused = degs[i_exp] * (adj @ (degs[j_exp] * z))
            res = torch.add(res, alpha=mult, other=diffused)

        if "att" in filt:
            assert att is not None
            values, edge_idx, degs = att
            mult, j_exp, i_exp = filt["att"]
            diffused = degs[j_exp] * z
            diffused = diffused[edge_idx[0]] * values
            diffused = degs[i_exp] * scatter(diffused, edge_idx[1], dim=0, dim_size=data.num_nodes)
            res = torch.add(res, alpha=mult, other=diffused)

        return res

    def forward(self, data):
        n = data.num_nodes
        one_hop_csr = data.one_hop_csr
        h1_deg_pows = data.h1_deg_pows

        alpha = self.gamma * F.softmax(self.alpha, dim=0)

        x = data.x
        x = self.layer1(x)  # Compute the embedding

        # add Gaussian noise to embeddings (for regularisation...)
        if self.noise_mult > 0 and self.training:
            x = x + self.noise_mult * torch.randn(x.shape, dtype=torch.float, device=self.device)

        # Split into channels, one per filter, each channel is of width self.hid_dim
        xs = list(torch.split(x, self.hid_dim, dim=1))

        # empty result tensor
        out = torch.zeros((n, self.lay_2_dim), dtype=torch.float, device=self.device)

        att = None  # attention matrix (usually not used)
        if self.att_layer is not None:
            att = self.att_layer(data)

        # Start with single-pass filters
        for i, z in enumerate(xs):
            filt = self.filters[i]

            # use the function above
            z = self.apply_filter(filt, z, data, att)

            z = F.normalize(z, p=2, dim=1)  # HOPNORM
            if self.leaky is not None:
                z = self.leaky(z)
            z = F.dropout(z, p=self.dp1, training=self.training)
            xs[i] = z
            out = out + alpha[i] * F.normalize(self.out_layers[i](z))

        # 2nd-propagation filters take the results of the starting filters and convolve them to neighbours again
        if self.double_prop_layer is not None:
            x_cat = torch.cat(xs, dim=1)
            assert x_cat.shape == (n, self.num_in_flts * self.hid_dim)
            double_prop = self.double_prop_layer(x_cat)
            double_prop = list(torch.split(double_prop, self.hid_dim, dim=1))
            for i, filt in enumerate(self.prop2_filts):
                z_ego = double_prop[i]
                z = h1_deg_pows[filt[2]] * (one_hop_csr @ (h1_deg_pows[filt[1]] * z_ego))
                if filt[0] != 0:
                    z = torch.add(z, alpha=filt[0], other=z_ego)
                z = F.normalize(z, p=2, dim=1)
                if self.leaky is not None:
                    z = self.leaky(z)
                z = F.dropout(z, p=self.dp1, training=self.training)
                out = out + alpha[i + self.num_in_flts] * F.normalize(self.out_layers[i](z))

        # A layer which can inject the degree of each node as a feature
        if self.deg_layer is not None:
            out = out + alpha[-1] * self.deg_layer(h1_deg_pows[1])

        # An additional two-layer MLP
        if self.decoder is not None:
            out = self.decoder(out)

        y_hat = F.log_softmax(out, dim=1)
        return y_hat
