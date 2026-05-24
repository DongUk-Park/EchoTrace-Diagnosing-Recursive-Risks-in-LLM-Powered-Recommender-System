import datetime
import math
import os
import random
import sys
from time import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.sparse as sparse

#from lattice_utility.parser import parse_args as lattice_parse_args
from LATTICE_Models import LightGCN
from lattice_utility.batch_test import *
from utility.parser import parse_args
lattice_args = parse_args()

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed_all(seed)  # GPU


class LightGCNTrainer(object):
    def __init__(self, data_config, data_generator):
        self.data_generator = data_generator
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.path = data_generator.path

        self.model_name = "lightgcn"
        self.lr = lattice_args.lr
        self.emb_dim = lattice_args.embed_size
        self.weight_size = eval(lattice_args.weight_size)
        self.n_layers = len(self.weight_size)
        self.regs = eval(lattice_args.regs)
        self.decay = self.regs[0]

        # 전체 normalized adjacency matrix
        norm_adj_sp = data_config['norm_adj'].tocoo().tocsr()
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(norm_adj_sp).float().to(self.device)

        # 사용자-아이템 인접행렬 분리
        n_users = self.n_users
        n_items = self.n_items
        ui_adj = norm_adj_sp[:n_users, n_users:]
        iu_adj = norm_adj_sp[n_users:, :n_users]

        self.ui_graph = self.sparse_mx_to_torch_sparse_tensor(ui_adj).float().to(self.device)
        self.iu_graph = self.sparse_mx_to_torch_sparse_tensor(iu_adj).float().to(self.device)

        self.model = LightGCN(
            n_users=self.n_users,
            n_items=self.n_items,
            embedding_dim=self.emb_dim,
            n_layers=self.n_layers,
            dropout=lattice_args.drop_rate,
            cat_rate=lattice_args.model_cat_rate,
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = self.set_lr_scheduler()

    def set_lr_scheduler(self):
        fac = lambda epoch: 0.96 ** (epoch / 50)
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)

    def train(self):
        print(f"🚀 LightGCN Trainer - Training started. user_num={self.n_users}, item_num={self.n_items}")
        
        for epoch in range(lattice_args.epoch):
            self.model.train()
            epoch_loss = 0.
            n_batch = self.data_generator.n_train // lattice_args.batch_size + 1
            for _ in range(n_batch):
                users, pos_items, neg_items = self.data_generator.sample()
                users = torch.tensor(users).to(self.device)
                pos_items = torch.tensor(pos_items).to(self.device)
                neg_items = torch.tensor(neg_items).to(self.device)

                ua_emb, ia_emb = self.model(self.ui_graph, self.iu_graph)
                u_g = ua_emb[users]
                pos_i_g = ia_emb[pos_items]
                neg_i_g = ia_emb[neg_items]

                mf_loss, emb_loss, reg_loss = self.bpr_loss(u_g, pos_i_g, neg_i_g)
                loss = mf_loss + emb_loss + reg_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            self.lr_scheduler.step()
            print(f"[Epoch {epoch}] Loss: {epoch_loss:.4f}")

            if epoch % lattice_args.verbose != 0:
                continue
            
        self.get_candidate()

    def get_candidate(self):
        self.model.eval()
        with torch.no_grad():
            ua_embeddings, ia_embeddings = self.model(self.ui_graph, self.iu_graph)
        # save top-k
        top_k = 10
        _, candidate_indices = torch.topk(torch.matmul(ua_embeddings, ia_embeddings.T), k=top_k, dim=1)
        save_path = os.path.join(self.path, 'candidate_indices')
        with open(save_path, 'wb') as f:
            pickle.dump(candidate_indices.cpu(), f)
        print(f"✅ Saved candidate indices to {save_path}")

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(users * pos_items, dim=1)
        neg_scores = torch.sum(users * neg_items, dim=1)

        reg_loss = 0.5 * (users.norm(2).pow(2) + pos_items.norm(2).pow(2) + neg_items.norm(2).pow(2)) / float(len(users))
        mf_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        emb_loss = self.decay * reg_loss
        return mf_loss, emb_loss, 0.0

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

def main(data_path, folder):
    set_seed(lattice_args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\U0001F4BB Using device: {device}")
    print(f"data path: {data_path}")
    data_generator = Data(path=os.path.join(data_path, folder), batch_size=lattice_args.batch_size)
    lattice_args.batch_size = 128
    lattice_args.lr = 1e-3
    lattice_args.de_lr = 4e-4 
    lattice_args.weight_decay = 1e-5  
    lattice_args.embed_size = 64
    lattice_args.layers = 3
    lattice_args.drop_rate = 0.0

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    _, norm_adj, _ = data_generator.get_adj_mat()
    config['norm_adj'] = norm_adj

    trainer = LightGCNTrainer(config, data_generator)
    trainer.train()

if __name__ == '__main__':
    main()