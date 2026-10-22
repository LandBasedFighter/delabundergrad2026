import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch.nn.functional as F
from tqdm import tqdm
from torch import optim
from torch.optim import Adam
from deeprobust.graph import utils
from torch_geometric.utils import to_dense_adj
from copy import deepcopy

##### GCN
class MyGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.norm = gcn_norm
        self.conv1 = GCNConv(in_channels, hidden_channels, normalize=False)
        self.conv2 = GCNConv(hidden_channels, out_channels, normalize=False)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, **kwargs):
        # Normalize edge indices only once
        if not kwargs.get('skip_norm', False):
            edge_index, edge_weight = self.norm(
                edge_index,
                edge_weight,
                num_nodes=x.size(0),
                add_self_loops=True,
            )

        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x
    
    def fit_with_val(self, features, adj, labels, data, train_iters=200, verbose=False, normalize=True, patience=None, noval=False, **kwargs):
        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)
        
        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj

        if 'feat_norm' in kwargs and kwargs['feat_norm']:
            from utils import row_normalize_tensor
            features = row_normalize_tensor(features-features.min())

        self.adj_norm = adj_norm
        self.features = features

        edge_index = adj_norm.nonzero(as_tuple=False).t()
        edge_weight = adj_norm[edge_index[0], edge_index[1]]

        if len(labels.shape) > 1:
            self.multi_label = True
            self.loss = torch.nn.BCELoss()
        else:
            self.multi_label = False
            self.loss = F.nll_loss

        labels = labels.float() if self.multi_label else labels
        self.labels = labels

        if noval:
            self._train_with_val(features, edge_index, edge_weight, labels, data, train_iters, adj_val=True)
        else:
            self._train_with_val(features, edge_index, edge_weight, labels, data, train_iters)

    def _train_with_val(self, features, edge_index, edge_weight, labels, data, train_iters, adj_val=False):
        if adj_val:
            feat_full, adj_full = data.feat_val, data.adj_val
        else:
            feat_full, adj_full = data.x, data.adj

        feat_full, adj_full = feat_full.to(self.device), adj_full.to(self.device)
        adj_full_norm = utils.normalize_adj_tensor(adj_full, sparse=True)
        edge_index_full = adj_full_norm.nonzero(as_tuple=False).t()
        edge_weight_full = adj_full_norm[edge_index_full[0], edge_index_full[1]]
        labels_val = torch.LongTensor(data.labels_val).to(self.device)
        
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_acc_val = 0
        for i in range(train_iters):
            if i == train_iters // 2:
                lr = self.lr*0.1
                optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)

            self.train()
            optimizer.zero_grad()
            output = self.forward(features, edge_index, edge_weight)
            loss_train = self.loss(output, labels)
            loss_train.backward()
            optimizer.step()

            with torch.no_grad():
                self.eval()
                # output = self.forward(feat_full, adj_full_norm)
                output = self.forward(feat_full, edge_index_full, edge_weight_full)
                if adj_val:
                    acc_val = utils.accuracy(output, labels_val)
                else:
                    acc_val = utils.accuracy(output[data.idx_val], labels_val)
                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    self.output = output
                    weights = deepcopy(self.state_dict())

        self.load_state_dict(weights)



