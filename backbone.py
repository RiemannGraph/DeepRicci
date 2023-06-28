import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, GraphAttentionLayer, SpGraphAttentionLayer
from torch_geometric.nn import GCNConv, SAGEConv
from utils import graph_top_K


class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, n_layers, dropout_node=0.5, dropout_edge=0.25):
        super(GCN, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GraphConvolution(in_features, hidden_features))
        for _ in range(n_layers - 2):
            self.conv_layers.append(GraphConvolution(hidden_features, hidden_features))
        self.conv_layers.append(GraphConvolution(hidden_features, out_features))
        self.dropout_node = nn.Dropout(dropout_node)
        self.dropout_edge = nn.Dropout(dropout_edge)

    def forward(self, x, adj):
        adj = self.dropout_edge(adj)
        for layer in self.conv_layers[: -1]:
            x = layer(x, adj)
            x = self.dropout_node(F.relu(x))
        x = self.conv_layers[-1](x, adj)
        return x


class GAT(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout_node=0.5, dropout_edge=0.25, alpha=0.2,
                 n_heads=4):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout_node
        self.dropout_edge = nn.Dropout(dropout_edge)

        self.attentions = [
            GraphAttentionLayer(in_features, hidden_features, dropout=dropout_node, alpha=alpha, concat=True) for _ in
            range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(hidden_features * n_heads, out_features, dropout=dropout_node, alpha=alpha,
                                           concat=False)

    def forward(self, x, adj):
        adj = self.dropout_edge(adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        return x


class SpGAT(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout_node=0.5, dropout_edge=0.25, alpha=0.2,
                 n_heads=4):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout_node
        self.dropout_edge = nn.Dropout(dropout_edge)

        self.attentions = [SpGraphAttentionLayer(in_features,
                                                 hidden_features,
                                                 dropout=dropout_node,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(hidden_features * n_heads,
                                             out_features,
                                             dropout=dropout_node,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, x, adj):
        adj = self.dropout_edge(adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        return x


class GraphSAGE(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, n_layers, dropout_node=0.5, dropout_edge=0.25):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(SAGEConv(in_features, hidden_features))
        for _ in range(n_layers - 2):
            self.conv_layers.append(SAGEConv(hidden_features, hidden_features))
        self.conv_layers.append(SAGEConv(hidden_features, out_features))
        self.dropout_node = nn.Dropout(dropout_node)
        self.dropout_edge = nn.Dropout(dropout_edge)

    def forward(self, x, adj):
        adj = self.dropout_edge(adj)
        edge_index = adj.nonzero().t()
        for layer in self.conv_layers[: -1]:
            x = layer(x, edge_index)
            x = self.dropout_node(F.relu(x))
        x = self.conv_layers[-1](x, edge_index)
        return x

    # class GraphEncoder(nn.Module):


#     def __init__(self, n_layers, in_features, hidden_features, embed_features, dropout, dropout_edge):
#         super(GraphEncoder, self).__init__()
#         self.dropout_node = nn.Dropout(dropout)
#         self.dropout_adj = nn.Dropout(dropout_edge)

#         self.encoder_layers = nn.ModuleList()
#         self.encoder_layers.append(GraphConvolution(in_features, hidden_features))
#         for _ in range(n_layers - 2):
#             self.encoder_layers.append(GraphConvolution(hidden_features, hidden_features))
#         self.encoder_layers.append(GraphConvolution(hidden_features, embed_features))

#     def forward(self, x, adj):
#         adj = self.dropout_adj(adj)
#         for layer in self.encoder_layers[:-1]:
#             x = self.dropout_node(F.relu(layer(x, adj)))
#         x = self.encoder_layers[-1](x, adj)
#         return x


class GraphEncoder(nn.Module):
    def __init__(self, backbone, n_layers, in_features, hidden_features, embed_features,
                 dropout, dropout_edge, alpha=0.2, n_heads=4, topk=30):
        super(GraphEncoder, self).__init__()
        if backbone == 'gcn':
            model = GCN(in_features, hidden_features, embed_features, n_layers,
                        dropout, dropout_edge)
        elif backbone == 'sage':
            model = GraphSAGE(in_features, hidden_features, embed_features, n_layers,
                              dropout, dropout_edge)
        elif backbone == 'gat':
            model = GAT(in_features, hidden_features, embed_features,
                        dropout, dropout_edge,
                        alpha, n_heads)
        elif backbone == 'spgat':
            model = SpGAT(in_features, hidden_features, embed_features,
                          dropout, dropout_edge,
                          alpha, n_heads)
        else:
            raise NotImplementedError

        self.backbone = backbone
        self.model = model
        self.topk = topk

    def forward(self, x, adj):
        if self.backbone in ['gat', 'spgat', 'sage']:
            adj = graph_top_K(adj, self.topk)
        return self.model(x, adj)