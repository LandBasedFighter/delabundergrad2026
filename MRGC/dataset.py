import os
import numpy as np
import torch
import numpy as np
# from deeprobust.graph.utils import *
try:
    from torch_geometric.loader import NeighborSampler
except ImportError:
    from torch_geometric.data import NeighborSampler
from torch_geometric.utils import to_dense_adj
import scipy.sparse as sp
import deeprobust.graph.utils as utils
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
from tqdm import tqdm

def edge_index_to_sp(edge_index, num_nodes):
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    data = [1] * len(row)
    adj_matrix = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    return adj_matrix


CLEAN_ROOT = './data/clean_source'


def load_clean_source(name, idx_train):
    """Clean features, edges and labels from the dataset's public source (downloaded on first use).

    Sizes match the shipped attacked files: Planetoid Cora/CiteSeer/PubMed, CitationFull DBLP
    (17716 x 1639, 4 classes) and ogbn-arxiv (undirected, features standardised on the train split).
    """
    from torch_geometric.utils import to_undirected
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        from torch_geometric.datasets import Planetoid
        g = Planetoid(os.path.abspath(CLEAN_ROOT), name)[0]
        x = g.x
    elif name == 'dblp':
        from torch_geometric.datasets import CitationFull
        g = CitationFull(os.path.abspath(CLEAN_ROOT), 'DBLP')[0]
        x = g.x
    elif name == 'ogbn-arxiv':
        from ogb.nodeproppred import PygNodePropPredDataset
        from sklearn.preprocessing import StandardScaler
        g = PygNodePropPredDataset(name='ogbn-arxiv', root=os.path.abspath(CLEAN_ROOT))[0]
        feat = g.x.numpy()
        feat = StandardScaler().fit(feat[idx_train]).transform(feat)
        x = torch.from_numpy(feat).float()
    else:
        raise NotImplementedError(f'No clean source for {name}')
    edge_index = to_undirected(g.edge_index, num_nodes=g.num_nodes)
    return {'f': x.float(), 's': edge_index.long(), 'l': g.y.view(-1).long()}


def check_alignment(name, attack_type, budget, clean, attacked):
    """Fail loudly if the rebuilt clean data does not match the attacked copy's node order.

    An attack at budget b should leave at least about (1 - b) of the rows / labels / edges untouched.
    """
    if attack_type == 'f':
        if clean.shape != attacked.shape:
            raise RuntimeError(f'{name}: clean features {tuple(clean.shape)} vs attacked {tuple(attacked.shape)}')
        same = ((clean - attacked.float()).abs().max(dim=1).values < 1e-4).float().mean().item()
        what = 'feature rows identical'
    elif attack_type == 'l':
        if clean.shape != attacked.view(-1).shape:
            raise RuntimeError(f'{name}: clean labels {tuple(clean.shape)} vs attacked {tuple(attacked.shape)}')
        same = (clean == attacked.view(-1).long()).float().mean().item()
        what = 'labels identical'
    else:
        n = int(max(clean.max(), attacked.max())) + 1
        key_clean = torch.unique(clean[0] * n + clean[1])
        key_att = torch.unique(attacked[0].long() * n + attacked[1].long())
        same = torch.isin(key_clean, key_att).float().mean().item()
        what = 'clean edges kept'
    print(f'[clean {attack_type}] {name}: {same:.1%} {what} vs attacked copy (budget {budget})')
    if same < 1 - budget - 0.1:
        raise RuntimeError(f'{name}: only {same:.1%} {what}; the clean source does not line up with the '
                           f'attacked files (different node order or preprocessing). Not running.')


def to_tensor(adj):
    if sp.issparse(adj):
        adj = utils.sparse_mx_to_torch_sparse_tensor(adj)
    else:
        adj = torch.FloatTensor(adj)
    return adj


class Dataset:
    def __init__(self, args):
        # ==================== Load ====================
        self.args = args
        data = self.load_dataset()
        # ==================== Process ====================
        if args.transductive:
            idx_train, idx_val, idx_test = data['idx_train'].numpy(), data['idx_val'].numpy(), data['idx_test'].numpy()
            # --feature_attack/--structure_attack/--label_attack 0 swaps in the clean component
            x = self.load_component('f', args.feature_budget, args.feature_attack, idx_train)
            edge_index = self.load_component('s', args.structure_budget, args.structure_attack, idx_train)
            y = self.load_component('l', args.label_budget, args.label_attack, idx_train)
            adj = edge_index_to_sp(edge_index, num_nodes=x.shape[0])

            nfeat = data['num_features']
            nclass = data['num_classes']

            # ==================== Outlier ====================
            if args.feature_outlier:
                x_train = x[idx_train].numpy()
                y_train = y[idx_train].numpy()
                outlier = []
                # print(x_train.shape)
                dist = cdist(x_train, x_train, metric='euclidean')
                for c in range(nclass):
                    c_idx = np.where(y_train == c)[0]
                    c_dist = dist[c_idx][:, c_idx]
                    c_dist = np.mean(c_dist, axis=1)
                    c_dist_mean = np.mean(c_dist)
                    c_dist_std = np.std(c_dist)
                    idx_outlier = c_idx[c_dist > c_dist_mean + args.feature_outlier_scale * c_dist_std]
                    # idx_outlier = idx_train[idx_outlier]
                    outlier.extend(idx_outlier)
                self.outlier = np.array(outlier)

                # real_outlier = np.where(np.any(data['x'][idx_train].numpy() != x[idx_train].numpy(), axis=1))[0]
                # print('----------------------------------')
                # print(sorted(list(real_outlier)))
                # print(sorted(list(outlier)))
                # print('----------------------------------')


            # # ==================== Normalize ====================
            # if args.normalize_features:
            #     x = x / x.sum(dim=1, keepdim=True).clamp(min=1e-12)
            #     x_test = x_test / x_test.sum(dim=1, keepdim=True).clamp(min=1e-12)


            # ==================== Assign ====================
            self.x = self.x_full = x.numpy()
            self.adj = self.adj_full = adj
            self.y = y.numpy()
            self.idx_train = idx_train
            self.idx_val = idx_val
            self.idx_test = idx_test

            self.labels_train = y[idx_train].cpu().numpy()
            self.labels_val = y[idx_val].cpu().numpy()
            self.labels_test = y[idx_test].cpu().numpy()
            
            self.nfeat = nfeat
            self.nclass = nclass


        self.device = args.device
        self.class_dict = None
        self.samplers = None
        self.class_dict2 = None

    
    def load_component(self, attack_type, budget, attacked, idx_train):
        attacked_value = self.load_attacked_dataset(attack_type, budget)
        if attacked:
            return attacked_value
        # The shipped data has no clean x/edges/labels (clean.pth only holds splits and sizes),
        # so rebuild them from the public source and check they line up with the attacked copy
        clean = self.load_clean_source(idx_train)[attack_type]
        check_alignment(self.args.name, attack_type, budget, clean, attacked_value)
        return clean

    def load_clean_source(self, idx_train):
        if getattr(self, '_clean', None) is None:
            self._clean = load_clean_source(self.args.name, idx_train)
        return self._clean

    def load_attacked_dataset(self, attack_type, budget):
        if attack_type == 's':
            save = torch.load(f'./data/{self.args.name}/s_{budget}.pth', map_location='cpu')
            return save['edge_index_attacked']
        elif attack_type == 'f':
            save = torch.load(f'./data/{self.args.name}/f_{budget}.pth', map_location='cpu')
            return save['x_attacked']
        elif attack_type == 'l':
            save = torch.load(f'./data/{self.args.name}/l_{budget}.pth', map_location='cpu')
            return save['y_attacked']
        else:
            raise FileNotFoundError
        
    def load_dataset(self):
        dataset = torch.load(os.path.abspath(f'./data/{self.args.name}/clean.pth'), map_location='cpu')
        return dataset


    def retrieve_class(self, c, num=256):
        if self.class_dict is None:
            self.class_dict = {}
            for i in range(self.nclass):
                self.class_dict['class_%s'%i] = (self.labels_train == i)
        idx = np.arange(len(self.labels_train))
        idx = idx[self.class_dict['class_%s'%c]]

        if self.args.feature_outlier:
            idx = np.setdiff1d(idx, self.outlier)
            if len(idx) >= num:
                idx_selected = np.random.permutation(idx)[:num]
            else:
                idx_selected = np.random.permutation(idx)
                idx_selected = np.concatenate([idx_selected, np.random.choice(idx_selected, num - len(idx_selected), replace=True)])
            return idx_selected
        else:
            return np.random.permutation(idx)[:num]
        
    def update_class_dict(self, transductive):
        # self.y_error_idx = {}
        if self.class_dict2 is None:
            self.class_dict2 = {}
            for i in range(self.nclass):
                if transductive:
                    idx = self.idx_train[self.labels_train == i]
                else:
                    idx = np.arange(len(self.labels_train))[self.labels_train==i]
                    
                self.class_dict2[i] = idx

    def retrieve_class_sampler(self, c, adj, transductive, num=256, args=None):
        if self.class_dict2 is None:
            self.class_dict2 = {}
            for i in range(self.nclass):
                if transductive:
                    idx = self.idx_train[self.labels_train == i]
                else:
                    idx = np.arange(len(self.labels_train))[self.labels_train==i]
                self.class_dict2[i] = idx

        if args.nlayers == 1:
            sizes = [15]
        if args.nlayers == 2:
            sizes = [10, 5]
        if args.nlayers == 3:
            sizes = [15, 10, 5]
        if args.nlayers == 4:
            sizes = [15, 10, 5, 5]
        if args.nlayers == 5:
            sizes = [15, 10, 5, 5, 5]


        if self.samplers is None:
            self.samplers = []
            for i in range(self.nclass):
                # node_idx = torch.LongTensor(self.class_dict2[i])
                node_idx = torch.tensor(self.class_dict2[i], dtype=torch.int64)
                if len(node_idx) == 0:
                    continue
                self.samplers.append(NeighborSampler(adj,
                                    node_idx=node_idx,
                                    sizes=sizes, batch_size=num,
                                    num_workers=2, return_e_id=False,
                                    num_nodes=adj.size(0),
                                    shuffle=True))
        batch = torch.tensor(np.random.permutation(self.class_dict2[c])[:num], dtype=torch.int64)
        out = self.samplers[c].sample(batch)
        return out

    def retrieve_class_multi_sampler(self, c, adj, transductive, num=256, args=None):
        if self.class_dict2 is None:
            self.class_dict2 = {}
            for i in range(self.nclass):
                if transductive:
                    idx = self.idx_train[self.labels_train == i]
                else:
                    idx = np.arange(len(self.labels_train))[self.labels_train==i]
                self.class_dict2[i] = idx


        if self.samplers is None:
            self.samplers = []
            for l in range(2):
                layer_samplers = []
                sizes = [15] if l == 0 else [10, 5]
                for i in range(self.nclass):
                    node_idx = torch.LongTensor(self.class_dict2[i])
                    layer_samplers.append(NeighborSampler(adj,
                                        node_idx=node_idx,
                                        sizes=sizes, batch_size=num,
                                        num_workers=12, return_e_id=False,
                                        num_nodes=adj.size(0),
                                        shuffle=True))
                self.samplers.append(layer_samplers)
        batch = np.random.permutation(self.class_dict2[c])[:num]
        out = self.samplers[args.nlayers-1][c].sample(batch)
        return out
