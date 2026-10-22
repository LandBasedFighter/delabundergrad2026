import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import match_loss
import deeprobust.graph.utils as utils
import numpy as np

from tqdm import tqdm
from models.gcn import GCN
from models.sgc import SGC
from models.parametrized_adj import PGE
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from torch_sparse import SparseTensor
from tools import *




class Trainer:
    def __init__(self, data, args, device='cuda:0', **kwargs):
        self.data = data
        self.args = args
        self.device = device

        n = int(data.x[data.idx_train].shape[0] * args.reduction_rate)

        self.nnodes_syn = n
        self.feat_syn = nn.Parameter(torch.FloatTensor(n, data.nfeat).to(device))
        self.pge = PGE(nfeat=data.nfeat, nnodes=n, device=device, args=args).to(device)
        self.labels_syn = torch.LongTensor(self.generate_labels_syn(data)).to(device)
        self.reset_parameters()
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=args.lr_adj)
        print('adj_syn:', (n,n), 'feat_syn:', self.feat_syn.shape)

    def reset_parameters(self):
        self.feat_syn.data.copy_(torch.randn(self.feat_syn.size()))
    
    def generate_labels_syn(self, data):
        num_classes = len(set(data.labels_train)) 
        n = len(data.labels_train)
        target_total_labels = int(n * self.args.reduction_rate)
        labels_per_class = target_total_labels // num_classes
        remainder = target_total_labels % num_classes
        labels_syn = []
        self.syn_class_indices = {}
        sum_ = 0
        for ix, c in enumerate(set(data.labels_train)):
            if ix < remainder:
                additional = 1
            else:
                additional = 0
            total_labels = labels_per_class + additional
            self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + total_labels]
            labels_syn += [c] * total_labels
            sum_ += total_labels
        self.num_class_dict = {c: labels_per_class + (1 if i < remainder else 0) for i, c in enumerate(set(data.labels_train))}
        return labels_syn

    def test_with_val(self):
        ### Data
        data, device = self.data, self.device
        feat_syn, pge, labels_syn = self.feat_syn.detach(), self.pge, self.labels_syn
        adj_syn = pge.inference(feat_syn)
        if self.args.lr_adj == 0:
            n = len(labels_syn)
            adj_syn = torch.zeros((n, n))
        
        ### Model
        model = GCN(nfeat=feat_syn.shape[1], nhid=self.args.hidden, dropout=0.5,
                    weight_decay=5e-4, nlayers=2,
                    nclass=data.nclass, device=device).to(device)
        if self.args.name in ['ogbn-arxiv']:
            model = GCN(nfeat=feat_syn.shape[1], nhid=self.args.hidden, dropout=0.5,
                        weight_decay=0e-4, nlayers=2, with_bn=False,
                        nclass=data.nclass, device=device).to(device)
        model.fit_with_val(feat_syn, adj_syn, labels_syn, data, train_iters=600, normalize=True, verbose=False)
        model.eval()
        
        ### Result
        labels_val = torch.LongTensor(data.labels_val).to(self.device)
        labels_test = torch.LongTensor(data.labels_test).to(self.device)
        # output = model.predict(data.x_test, data.adj_test)
        output = model.predict(data.x, data.adj)
        acc_val = utils.accuracy(output[data.idx_val], labels_val)
        acc_test = utils.accuracy(output[data.idx_test], labels_test)
        return acc_val, acc_test

    def get_sub_adj_feat(self, features):
        data = self.data
        idx_selected = []

        from collections import Counter;
        counter = Counter(self.labels_syn.cpu().numpy())
        for c in range(data.nclass):
            tmp = data.retrieve_class(c, num=counter[c]) # the index of the class c
            tmp = list(tmp)
            idx_selected = idx_selected + tmp
        idx_selected = np.array(idx_selected).reshape(-1)
        features = features[self.data.idx_train][idx_selected]
        return features
    
    def manifold_dimension(self, x, k):
        adj = neighbor_graph(x, k=k)
        lap = laplacian(adj)
        dim = torch.trace(x.T @ lap @ x)
        return dim
    
    
    def train(self):
        args = self.args
        data = self.data
        syn_class_indices = self.syn_class_indices
        
        features, adj, labels = data.x, data.adj, data.y
        features, adj, labels = utils.to_tensor(features, adj, labels, device=args.device)

        feat_syn, pge, labels_syn = self.feat_syn, self.pge, self.labels_syn
        feat_sub = self.get_sub_adj_feat(features)
        self.feat_syn.data.copy_(feat_sub)

        adj = utils.normalize_adj_tensor(adj, sparse=True)
        adj = SparseTensor(row=adj._indices()[0], col=adj._indices()[1], value=adj._values(), sparse_sizes=adj.size()).t()
        
        ### Train
        outer_loop, inner_loop = args.outer, args.inner
        data.update_class_dict(transductive=True)
        for it in tqdm(range(args.epochs+1)):
            ### Model
            model = SGC(nfeat=data.x.shape[1], nhid=args.hidden,
                        nclass=data.nclass, dropout=args.dropout,
                        nlayers=args.nlayers, with_bn=False,
                        device=self.device).to(self.device)

            model.initialize()
            model_parameters = list(model.parameters())
            optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr_model)
            model.train()

            
            for ol in range(outer_loop):
                loss, loss_dimension, loss_curvature = 0., 0., 0.

                adj_syn = pge(self.feat_syn)
                adj_syn_norm = utils.normalize_adj_tensor(adj_syn)
                
                prop_feat = adj_syn_norm @ (adj_syn_norm @ feat_syn)
                try:
                    prop_feat = pca_svd(prop_feat, k=args.prop_dim)
                except:
                    print(torch.isnan(feat_syn).any())
                    print(torch.isnan(adj_syn).any())
                    print(torch.isnan(adj_syn_norm).any())
                    print(torch.isnan(prop_feat).any())
                    exit()

                # Total Volume
                volume_list = []
                total_volume = calculate_volume(prop_feat)
                
                # Ricci Curvature
                if it % 50 == 0:
                    with torch.no_grad():
                        node_ricci = compute_ricci_curvature(adj_syn_norm, alpha=0.5)
                
                output_syn = model.forward(feat_syn, adj_syn_norm)
                for c in range(data.nclass):
                    # ==================== Gradient Matching ====================
                    batch_size, n_id, adjs = data.retrieve_class_sampler(c, adj, transductive=True, args=args)
                    if args.nlayers == 1: adjs = [adjs]
                    adjs = [adj.to(self.device) for adj in adjs]
                    output = model.forward_sampler(features[n_id], adjs)
                    labels_idx = labels[n_id[:batch_size]]
                    loss_real = F.nll_loss(output, labels_idx)
                    gw_real = torch.autograd.grad(loss_real, model_parameters, retain_graph=True)

                    ind = syn_class_indices[c]
                    loss_syn = F.nll_loss(output_syn[ind[0]: ind[1]], labels_syn[ind[0]: ind[1]])
                    gw_syn = torch.autograd.grad(loss_syn, model_parameters, create_graph=True)

                    coeff = self.num_class_dict[c] / max(self.num_class_dict.values())
                    loss_match = match_loss(gw_syn, gw_real, args, device=self.device)
                    
                    
                    # ==================== Intrinsic Dimension Manifold Regularization ====================
                    if ind[1] - ind[0] >= args.dimension_k + 1:
                        loss_dimension = total_volume * self.manifold_dimension(prop_feat[ind[0]: ind[1]], k=args.dimension_k)
                    # ==================== Curvature-Aware Manifold Smoothing ====================
                    if ind[1] - ind[0] >= args.curvature_k + 1:
                        node_gaussian_c = torch.abs(compute_gaussian_curvature_batch(prop_feat[ind[0]: ind[1]], k=args.curvature_k))
                        node_ricci_c = -node_ricci[ind[0]: ind[1]]
                        node_ricci_c = (node_ricci_c - torch.min(node_ricci_c)) / (torch.max(node_ricci_c) - torch.min(node_ricci_c))
                        loss_curvature = torch.sum(node_ricci_c * node_gaussian_c)
                    # ==================== Class-Wise Manifold Decoupling ====================
                        volume_c = calculate_volume(prop_feat[ind[0]: ind[1]])
                        volume_list.append(volume_c)

                    loss += coeff * (loss_match + args.alpha * loss_dimension + args.beta * loss_curvature)

                # ==================== Class-Wise Manifold Decoupling ====================          
                loss_volume = 0.
                loss_volume = torch.pow(torch.sum(torch.stack(volume_list), dim=0) - total_volume, 2)
                loss += args.gamma * loss_volume
                self.optimizer_feat.zero_grad()
                self.optimizer_pge.zero_grad()
                loss.backward()

                if it < 50:
                    if it % 50 < 40: self.optimizer_feat.step()
                    else: self.optimizer_pge.step()
                else:
                    if it % 50 < 10: self.optimizer_pge.step()
                    else: self.optimizer_feat.step()
                if ol == outer_loop - 1:
                    break

                feat_syn_inner = feat_syn.detach()
                adj_syn_inner = pge.inference(feat_syn_inner)
                adj_syn_inner_norm = utils.normalize_adj_tensor(adj_syn_inner)
                feat_syn_inner_norm = feat_syn_inner
                for _ in range(inner_loop):
                    optimizer_model.zero_grad()
                    output_syn_inner = model.forward(feat_syn_inner_norm, adj_syn_inner_norm)
                    loss_syn_inner = F.nll_loss(output_syn_inner, labels_syn)
                    loss_syn_inner.backward()
                    optimizer_model.step()
            
            if (it + 1) % 100 == 0:
                acc_val_res = []
                acc_test_res = []
                if args.name in ['ogbn-arxiv']: runs = 1
                else: runs = 3
                for i in tqdm(range(runs)):
                    acc_val, acc_test = self.test_with_val()
                    acc_val_res.append(acc_val.item())
                    acc_test_res.append(acc_test.item())
                acc_val_res = np.array(acc_val_res)
                acc_test_res = np.array(acc_test_res)

                # print(f' Val Accuracy Mean {acc_val_res.mean(0).item():.4f}, Std {acc_val_res.std(0).item():.4f}\nTest Accuracy Mean {acc_test_res.mean(0).item():.4f}, Std {acc_test_res.std(0).item():.4f}')
                # with open(args.log_file, 'a+') as f:
                #     f.write(f'Epoch {(it + 1):4d}\nVal Accuracy Mean {acc_val_res.mean(0).item():.4f}, Std {acc_val_res.std(0).item():.4f}\nTest Accuracy Mean {acc_test_res.mean(0).item():.4f}, Std {acc_test_res.std(0).item():.4f}\n')
