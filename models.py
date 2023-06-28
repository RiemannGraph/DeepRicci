import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import GraphEncoder
import numpy as np
from utils import normalize, graph_top_K, adjacency2index, graph_threshold
from utils import gumbel_sigmoid
from geoopt.manifolds.stereographic.math import artan_k
from geoopt.manifolds.stereographic import StereographicExact
from geoopt import ManifoldTensor
from geoopt import ManifoldParameter
from geoopt.manifolds.stereographic.math import logmap0


class RiemannianEmbeds(nn.Module):
    def __init__(self, num_nodes, d_riemann, cur_h=-1., cur_s=1.):
        super(RiemannianEmbeds, self).__init__()
        self.hyperbolic = StereographicExact(k=cur_h)
        self.sphere = StereographicExact(k=cur_s)
        self.embeds_hyperbolic = ManifoldParameter(ManifoldTensor(torch.empty(num_nodes, d_riemann),
                                                                  manifold=self.hyperbolic))  # N, D_r
        self.init_weights(self.embeds_hyperbolic)

        self.embeds_sphere = ManifoldParameter(ManifoldTensor(torch.empty(num_nodes, d_riemann),
                                                              manifold=self.sphere))  # N, D_r
        self.init_weights(self.embeds_sphere)

    def init_weights(self, w, scale=1e-4):
        w.data.uniform_(-scale, scale)
        w_norm = w.data.norm(p=2, dim=-1, keepdim=True)
        w.data = w.data / w_norm * w.manifold.radius * 0.9 * torch.rand(1)

    def normalize(self, x, manifold):
        x_norm = x.norm(p=2, dim=-1, keepdim=True)
        x = x / x_norm * 0.9 * torch.rand(1).to(x.device) * manifold.radius
        return x

    def forward(self):
        embeds_hyperbolic = self.normalize(self.embeds_hyperbolic, self.hyperbolic)
        embeds_sphere = self.normalize(self.embeds_sphere, self.sphere)
        return embeds_hyperbolic, embeds_sphere


class DeepRicci(nn.Module):
    def __init__(self, backbone, n_layers, n_heads, in_features, d_riemann,
                 d_hyla, hidden_features, embed_features, dropout,
                 dropout_edge, init_adj=None, scale=0.1, cur_h=-1., cur_s=1.,
                 s=1, r=2, gamma=1., top_k=30, eps=None,
                 alpha=0.5, backbone_topk=30, act='relu', wq='linear', act_adj='gumbel',
                 temperature=0.1, alpha_gat=0.2, n_heads_gat=8, device=torch.device('cuda')):
        super(DeepRicci, self).__init__()
        self.n_heads = n_heads
        self.act_adj = act_adj
        if wq == 'linear':
            self.w_q = nn.Linear(3 * d_hyla, embed_features * n_heads)
        else:
            activation = nn.ReLU() if act == 'relu' else nn.Tanh()
            self.w_q = nn.Sequential(
                nn.Linear(3 * d_hyla, hidden_features),
                activation,
                nn.Linear(hidden_features, embed_features * n_heads)
            )
        self.scale = scale
        self.alpha = alpha
        self.s = s
        self.r = r

        pre = torch.randn(d_hyla, in_features).to(device)
        self.boundary_matrix_eu = pre / torch.norm(pre, dim=-1, keepdim=True)  # D_h, in_feats
        self.bias_eu = 2 * torch.pi * torch.rand(d_hyla).to(device)  # n, D_h

        pre = torch.randn(d_hyla, d_riemann).to(device)
        self.boundary_matrix_h = pre / torch.norm(pre, dim=-1, keepdim=True)  # D_h, D_r
        self.eigen_values_h = torch.randn(d_hyla).to(device) * self.scale  # n, D_h
        self.bias_h = 2 * torch.pi * torch.rand(d_hyla).to(device)  # n, D_h

        pre = torch.randn(d_hyla, d_riemann).to(device)
        self.boundary_matrix_s = pre / torch.norm(pre, dim=-1, keepdim=True)  # D_h, D_r
        self.eigen_values_s = torch.randn(d_hyla).to(device) * self.scale  # n, D_h
        self.bias_s = 2 * torch.pi * torch.rand(d_hyla).to(device)  # n, D_h

        self.encoder = GraphEncoder(backbone, n_layers, in_features, hidden_features, d_hyla, dropout, dropout_edge,
                                    alpha_gat, n_heads_gat, backbone_topk)
        self.lipschitz_func = nn.Linear(3 * d_hyla, 1)
        self.init_adj = init_adj.to(device)
        self.gamma = gamma
        self.cur_h = cur_h
        self.cur_s = cur_s
        self.top_k = top_k
        self.eps = eps
        self.temperature = temperature

    def forward(self, feature, A, ratio, riemann_embeds_getter):
        embeds_hyperbolic, embeds_sphere = riemann_embeds_getter()

        feature_euclidean = np.sqrt(2) * torch.cos(torch.matmul(feature, self.boundary_matrix_eu.t()) + self.bias_eu)
        feature_hyperbolic = \
            self.cal_laplacian_features(embeds_hyperbolic, self.cur_h, self.boundary_matrix_h, self.eigen_values_h,
                                        self.bias_h)
        feature_sphere = \
            self.cal_laplacian_features(embeds_sphere, self.cur_s, self.boundary_matrix_s, self.eigen_values_s,
                                        self.bias_s)
        product_features = torch.concat([feature_euclidean, feature_hyperbolic, feature_sphere], dim=-1)

        learned_A = self.learn_adjacency(product_features)
        learned_A_normed = (1 - ratio) * normalize(learned_A, "sym") + ratio * normalize(A, "sym")
        learned_A_rownormed = (1 - ratio) * normalize(learned_A, "row") + ratio * normalize(A, "row")

        z_E = self.encoder(feature, learned_A_normed)
        product_z = torch.concat([z_E, embeds_hyperbolic, embeds_sphere], dim=-1)

        info_loss = self.cal_cl_loss(torch.concat([z_E, feature_hyperbolic], dim=-1),
                                     torch.concat([z_E, feature_sphere], dim=-1))

        structure_loss = self.cal_structure_loss(product_features, learned_A_rownormed, self.init_adj)
        loss = info_loss + self.gamma * structure_loss
        new_feature = torch.concat([feature, embeds_hyperbolic, embeds_sphere], dim=-1)
        return product_z, new_feature, learned_A_normed, loss

    def dist_to_horocycle(self, z, k, boundary_matrix):
        boundary_matrix = boundary_matrix / torch.norm(boundary_matrix, dim=-1, keepdim=True)
        div = 1 - torch.matmul(z, boundary_matrix.t())
        dist = (torch.matmul(z, boundary_matrix.t()) - torch.norm(z, dim=-1, keepdim=True) ** 2) / div
        P = artan_k(dist, torch.tensor(k))
        return P

    def cal_laplacian_features(self, z, k, boundary_matrix, eigen_values, bias):
        horocycle_dist = self.dist_to_horocycle(z, k, boundary_matrix)
        n = z.shape[-1]
        laplacian_features = torch.exp((n - 1) * horocycle_dist / 2) * \
                             torch.cos(eigen_values * horocycle_dist + bias)
        return laplacian_features

    def multi_head_attention(self, q, k):
        """
        q: [H, N, D]
        k: [H, N, D]
        """
        N = q.shape[0]
        M = k.shape[0]
        H = self.n_heads
        q = q.reshape(N, -1, H).permute(2, 0, 1)
        k = k.reshape(M, -1, H).permute(2, 0, 1)
        score = torch.einsum('hnd, hmd->hnm', q, k).mean(0)
        score = torch.softmax(score, dim=-1)
        return score

    def post_process(self, A, k=None, eps=None):
        # assert (k is not None) or (eps is not None)
        A = graph_top_K(A, k) if k is not None else graph_threshold(A, eps)
        return A

    def learn_adjacency(self, x):
        q = self.w_q(x)
        learned_A = self.multi_head_attention(q, q)
        learned_A = self.post_process(learned_A, self.top_k, self.eps)
        learned_A = (learned_A + learned_A.t()) / 2
        if self.act_adj == 'gumbel':
            learned_A = torch.sigmoid(learned_A)
            learned_A = gumbel_sigmoid(learned_A)
        elif self.act_adj == 'relu':
            learned_A = torch.relu(learned_A)
        elif self.act_adj == 'elu':
            learned_A = F.elu(6 * (learned_A - 1)) + 1
        return learned_A

    def compute_wasserstein_dist(self, x, adj, edge_index):
        """_summary_

        Args:
            x (_type_): node features
            adj (_type_): row normalized adjacency matrix
        # """
        src_idx = edge_index[0]
        tgt_idx = edge_index[1]
        f = self.lipschitz_func(x)
        f_x_src = f[src_idx]
        f_x_tgt = f[tgt_idx]
        L_bar = torch.eye(x.shape[0]).to(x.device) - (1 - 1 / self.alpha) * adj
        L_weight = L_bar[src_idx, tgt_idx]
        w_dist = self.alpha * L_weight * (f_x_src - f_x_tgt).reshape(-1)
        return w_dist.clip(min=0.)

    def compute_ricci_flow(self, x, adjacency, edge_index):
        """adjacency: row normalized"""
        w_dist_t = self.compute_wasserstein_dist(x, adjacency, edge_index)
        return torch.sigmoid((self.r - w_dist_t) / self.s)

    def cal_cl_loss(self, x1, x2):
        norm1 = x1.norm(dim=-1)
        norm2 = x2.norm(dim=-1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', norm1, norm2)
        sim_matrix = torch.exp(sim_matrix / self.temperature)
        pos_sim = sim_matrix.diag()
        loss_1 = pos_sim / (sim_matrix.sum(dim=-2) - pos_sim)
        loss_2 = pos_sim / (sim_matrix.sum(dim=-1) - pos_sim)

        loss_1 = -torch.log(loss_1).mean()
        loss_2 = -torch.log(loss_2).mean()
        loss = (loss_1 + loss_2) / 2.
        return loss

    def cal_structure_loss(self, x, learned_A_rownormed, A):
        edge_idx = adjacency2index(A, weight=False, topk=True, k=self.top_k)
        pos = self.compute_ricci_flow(x, learned_A_rownormed, edge_idx)
        loss = F.nll_loss(torch.log(pos + 1e-5), torch.ones_like(pos).long().to(pos.device))
        return loss