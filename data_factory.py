import torch
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import numpy as np
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor
from torch_geometric.utils import to_networkx
from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.datasets import load_wine, load_breast_cancer, load_digits, fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split


def get_mask(idx, length):
    """Create mask.
    """
    mask = torch.zeros(length, dtype=torch.bool)
    mask[idx] = 1
    return mask


def load_graph_data(root: str, data_name: str, split='public', **kwargs):
    if data_name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(root=root, name=data_name, split=split)
        train_mask, val_mask, test_mask = dataset.data.train_mask, dataset.data.val_mask, dataset.data.test_mask
    elif data_name == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(root=root, name=data_name)
        mask = dataset.get_idx_split()
        train_mask, val_mask, test_mask = mask.values()
    elif data_name in ['actor', 'chameleon', 'squirrel']:
        if data_name == 'actor':
            path = root + f'/{data_name}'
            dataset = Actor(root=path)
        else:
            dataset = WikipediaNetwork(root=root, name=data_name)
        num_nodes = dataset.data.x.shape[0]
        idx_train = []
        for j in range(dataset.num_classes):
            idx_train.extend([i for i, x in enumerate(dataset.data.y) if x == j][:20])
        idx_val = np.arange(num_nodes - 1500, num_nodes - 1000)
        idx_test = np.arange(num_nodes - 1000, num_nodes)
        label_len = dataset.data.y.shape[0]
        train_mask, val_mask, test_mask = get_mask(idx_train, label_len), get_mask(idx_val, label_len), get_mask(idx_test, label_len)
    else:
        raise NotImplementedError

    print(dataset.data)
    G = to_networkx(dataset.data)
    features = dataset.data.x
    num_features = dataset.num_features
    labels = dataset.data.y
    adjacency = torch.from_numpy(nx.adjacency_matrix(G).toarray())
    num_classes = dataset.num_classes
    return features, num_features, labels, adjacency, (train_mask, val_mask, test_mask), num_classes


def load_non_graph_data(root: str, data_name: str, seed=100, **kwargs):
    features = None
    if data_name == 'wine':
        dataset = load_wine()
        n_train = 10
        n_val = 10
        n_es = 10
        is_scale = True
    elif data_name == 'digits':
        dataset = load_digits()
        n_train = 50
        n_val = 50
        n_es = 50
        is_scale = False
    elif data_name == 'cancer':
        dataset = load_breast_cancer()
        n_train = 10
        n_val = 10
        n_es = 10
        is_scale = True
    elif data_name == '20news10':
        n_train = 100
        n_val = 100
        n_es = 100
        is_scale = False
        categories = ['alt.atheism',
                      'comp.sys.ibm.pc.hardware',
                      'misc.forsale',
                      'rec.autos',
                      'rec.sport.hockey',
                      'sci.crypt',
                      'sci.electronics',
                      'sci.med',
                      'sci.space',
                      'talk.politics.guns']
        dataset = fetch_20newsgroups(subset='all', categories=categories)
        vectorizer = CountVectorizer(stop_words='english', min_df=0.05)
        X_counts = vectorizer.fit_transform(dataset.data).toarray()
        transformer = TfidfTransformer(smooth_idf=False)
        features = transformer.fit_transform(X_counts).todense()
    else:
        raise NotImplementedError

    if data_name != '20news10':
        if is_scale:
            features = scale(dataset.data)
        else:
            features = dataset.data
    features = torch.from_numpy(features)
    y = dataset.target
    n, num_features = features.shape
    train, test, y_train, y_test = train_test_split(np.arange(n), y, random_state=seed,
                                                    train_size=n_train + n_val + n_es,
                                                    test_size=n - n_train - n_val - n_es,
                                                    stratify=y)
    train, es, y_train, y_es = train_test_split(train, y_train, random_state=seed,
                                                train_size=n_train + n_val, test_size=n_es,
                                                stratify=y_train)
    train, val, y_train, y_val = train_test_split(train, y_train, random_state=seed,
                                                  train_size=n_train, test_size=n_val,
                                                  stratify=y_train)

    train_mask = torch.zeros(n, dtype=bool)
    train_mask[train] = True
    val_mask = torch.zeros(n, dtype=bool)
    val_mask[val] = True
    es_mask = torch.zeros(n, dtype=bool)
    es_mask[es] = True
    test_mask = torch.zeros(n, dtype=bool)
    test_mask[test] = True
    labels = torch.from_numpy(y)
    num_classes = len(dataset.target_names)
    return features, num_features, labels, torch.zeros(n, n), (train_mask, val_mask, test_mask), num_classes


def load_data(args, **kwargs):
    if args.is_graph:
        data_getter = load_graph_data
    else:
        data_getter = load_non_graph_data
    return data_getter(args.root_path, args.dataset)