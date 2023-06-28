import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
import numpy as np
from sklearn import metrics
from munkres import Munkres
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def k_nearest_neighbors(x, k_neighbours, metric):
    adj = kneighbors_graph(x, k_neighbours, metric=metric)
    adj = adj.toarray().astype(np.float32)
    adj += np.eye(adj.shape[0])
    return adj


def graph_top_K(dense_adj, k):
    assert k < dense_adj.shape[-1]
    _, indices = dense_adj.topk(k=k, dim=-1)
    mask = torch.zeros(dense_adj.shape).bool().to(dense_adj.device)
    mask[torch.arange(dense_adj.shape[0])[:, None], indices] = True
    sparse_adj = torch.masked_fill(dense_adj, ~mask, value=0.)
    return sparse_adj


def graph_threshold(dense_adj, eps):
    sparse_adj = torch.masked_fill(dense_adj, (dense_adj < eps), value=0.)
    return sparse_adj


def cal_accuracy(preds, trues):
    preds = torch.argmax(preds, dim=-1)
    correct = (preds == trues).sum()
    return correct / len(trues)


def cal_F1(preds, trues):
    preds = torch.argmax(preds, dim=-1)
    weighted_f1 = metrics.f1_score(trues, preds, average='weighted')
    macro_f1 = metrics.f1_score(trues, preds, average='macro')
    return weighted_f1, macro_f1


def normalize(adj, mode, sparse=False):
    if sparse:
        adj = adj.coalesce()
        if mode == 'sym':
            degree_matrix = 1. / (torch.sqrt(torch.sparse.sum(adj, -1)))
            value = degree_matrix[adj.indices()[0]] * degree_matrix[adj.indices()[1]]
        elif mode == 'row':
            degree_matrix = 1. / (torch.sparse.sum(adj, -1))
            value = degree_matrix[adj.indices()[0]]
        else:
            raise NotImplementedError
        return torch.sparse_coo_tensor(adj.indices(), value * adj.values(), adj.shape)
    else:
        if mode == 'sym':
            degree_matrix = 1. / (torch.sqrt(adj.sum(-1)) + 1e-10)
            return degree_matrix[:, None] * adj * degree_matrix[None, :]
        elif mode == 'row':
            degree_matrix = 1. / (adj.sum(-1) + 1e-10)
        else:
            raise NotImplementedError
        return degree_matrix[:, None] * adj


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size())
    return torch.nn.functional.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=1, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


def adjacency2index(adjacency, weight=False, topk=False, k=10):
    """_summary_

    Args:
        adjacency (torch.tensor): [N, N] matrix
    return:
        edge_index: [2, E]
        edge_weight: optional
    """
    if topk and k:
        adj = graph_top_K(adjacency, k)
    else:
        adj = adjacency
    edge_index = torch.nonzero(adj).t().contiguous()
    if weight:
        weight = adjacency[edge_index[0], edge_index[1]].reshape(-1)
        return edge_index, weight

    else:
        return edge_index


def plot_graph(edges, labels, n_class):
    """_summary_

    Args:
        edges (ndarray): [E, 2]
        nodes (ndarray): [N, ]
        labels (ndarray): [N, ]
    """
    color_list = []
    colors = list(mcolors.CSS4_COLORS.keys())
    for i in range(n_class):
        color_code = random.choice(colors)
        color_list.append(color_code)
    print(f"Choose color list: {color_list}")
    G = nx.Graph()
    nodes = [(i, {'label': labels[i], 'color': color_list[labels[i]]}) for i in range(len(labels))]
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(edges)

    nodes_color = nx.get_node_attributes(G, 'color').values()
    edge_weights = nx.get_edge_attributes(G, 'weight')
    pos = nx.spring_layout(G)
    nx.draw(G, with_labels=True, node_color=nodes_color, pos=pos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights)


def curvature_arctanh(x, k):
    if k > 0:
        return (1 / np.sqrt(k)) * torch.arctanh(np.sqrt(k) * x)
    if k == 0:
        return x
    if k < 0:
        return (1 / np.sqrt(-k)) * torch.arctanh(np.sqrt(-k) * x)


def gumbel_sigmoid(probs, t=0.1, hard=False):
    eps = torch.rand_like(probs).to(probs.device)
    eps = eps.clip(0.01, 0.99)
    probs = probs.clip(0.001, 0.999)
    logits1 = probs.log() - (1 - probs).log()
    logits2 = eps.log() - (1 - probs).log()
    samples = torch.sigmoid((logits1 + logits2) / t)
    if hard:
        sampel_hard = (samples > 0.5).float()
        samples_hard = (sampel_hard - samples).detach() + samples
        return samples_hard
    return samples


class cluster_metrics:
    def __init__(self, trues, predicts):
        self.trues = trues
        self.predicts = predicts

    def clusterAcc(self):
        l1 = list(set(self.trues))
        l2 = list(set(self.predicts))
        num1 = len(l1)
        num2 = len(l2)
        if num1 != num2:
            raise Exception("number of classes not equal")

        """compute the cost of allocating c1 in L1 to c2 in L2"""
        cost = np.zeros((num1, num2), dtype=int)
        for i, c1 in enumerate(l1):
            maps = np.where(self.trues == c1)[0]
            for j, c2 in enumerate(l2):
                maps_d = [i1 for i1 in maps if self.predicts[i1] == c2]
                cost[i, j] = len(maps_d)

        mks = Munkres()
        index = mks.compute(-cost)
        new_predicts = np.zeros(len(self.predicts))
        for i, c in enumerate(l1):
            c2 = l2[index[i][1]]
            allocate_index = np.where(self.predicts == c2)[0]
            new_predicts[allocate_index] = c

        acc = metrics.accuracy_score(self.trues, new_predicts)
        f1_macro = metrics.f1_score(self.trues, new_predicts, average='macro')
        precision_macro = metrics.precision_score(self.trues, new_predicts, average='macro')
        recall_macro = metrics.recall_score(self.trues, new_predicts, average='macro')
        f1_micro = metrics.f1_score(self.trues, new_predicts, average='micro')
        precision_micro = metrics.precision_score(self.trues, new_predicts, average='micro')
        recall_micro = metrics.recall_score(self.trues, new_predicts, average='micro')
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro

    def evaluateFromLabel(self):
        nmi = metrics.normalized_mutual_info_score(self.trues, self.predicts)
        adjscore = metrics.adjusted_rand_score(self.trues, self.predicts)
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = self.clusterAcc()
        return acc, nmi, f1_macro, adjscore


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__