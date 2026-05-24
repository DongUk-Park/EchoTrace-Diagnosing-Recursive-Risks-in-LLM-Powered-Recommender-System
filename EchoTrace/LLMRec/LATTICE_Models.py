import os
import numpy as np
from time import time

import torch
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.functional as F

from lattice_utility.parser import parse_args

def build_knn_neighbourhood(adj, topk):
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
    return weighted_adjacency_matrix
def compute_normalized_laplacian(adj):
    rowsum = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return L_norm
def build_sim(context):
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    return sim

class LATTICE(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list, image_feats, text_feats):
        super().__init__()
        self.args = parse_args()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.weight_size = weight_size
        self.n_ui_layers = len(self.weight_size)
        self.weight_size = [self.embedding_dim] + self.weight_size
        self.user_embedding = nn.Embedding(n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        if self.args.cf_model == 'ngcf':
            self.GC_Linear_list = nn.ModuleList()
            self.Bi_Linear_list = nn.ModuleList()
            self.dropout_list = nn.ModuleList()
            for i in range(self.n_ui_layers):
                self.GC_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i+1]))
                self.Bi_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i+1]))
                self.dropout_list.append(nn.Dropout(dropout_list[i]))


        self.image_embedding = nn.Embedding.from_pretrained(torch.Tensor(image_feats), freeze=False)
        self.text_embedding = nn.Embedding.from_pretrained(torch.Tensor(text_feats), freeze=False)
            
    
        if os.path.exists(self.args.data_path + 'image_adj_%d.pt'%(self.args.topk)):
            image_adj = torch.load(self.args.data_path + 'image_adj_%d.pt'%(self.args.topk))
        else:
            image_adj = build_sim(self.image_embedding.weight.detach())
            image_adj = build_knn_neighbourhood(image_adj, topk=self.args.topk)
            image_adj = compute_normalized_laplacian(image_adj)
            torch.save(image_adj, self.args.data_path + 'image_adj_%d.pt'%(self.args.topk))

        if os.path.exists(self.args.data_path + 'text_adj_%d.pt'%(self.args.topk)):
            text_adj = torch.load(self.args.data_path + 'text_adj_%d.pt'%(self.args.topk))        
        else:
            text_adj = build_sim(self.text_embedding.weight.detach())
            text_adj = build_knn_neighbourhood(text_adj, topk=self.args.topk)
            text_adj = compute_normalized_laplacian(text_adj)
            torch.save(text_adj, self.args.data_path + 'text_adj_%d.pt'%(self.args.topk))

        self.text_original_adj = text_adj.cuda()
        #self.image_original_adj = image_adj.cuda()

        # Step 1: Make a 0-filled scipy sparse matrix
        import scipy.sparse as sp
        image_adj = sp.csr_matrix((n_items, n_items), dtype=np.float32)

        # Step 2: Convert to torch sparse tensor
        def scipy_to_torch_sparse(mat):
            mat = mat.tocoo()
            indices = torch.tensor(np.vstack((mat.row, mat.col)), dtype=torch.int64)
            values = torch.tensor(mat.data, dtype=torch.float32)
            shape = torch.Size(mat.shape)
            return torch.sparse_coo_tensor(indices, values, shape).to_dense()  # to_dense if needed

        image_adj_torch = scipy_to_torch_sparse(image_adj)

        # Step 3: CUDA 이동
        self.image_original_adj = image_adj_torch.cuda()

        
        self.image_trs = nn.Linear(image_feats.shape[1], self.args.feat_embed_dim)
        self.text_trs = nn.Linear(text_feats.shape[1], self.args.feat_embed_dim)


        self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, adj, build_item_graph=False):
        image_feats = self.image_trs(self.image_embedding.weight)
        text_feats = self.text_trs(self.text_embedding.weight)
        if build_item_graph:
            weight = self.softmax(self.modal_weight)
            self.image_adj = build_sim(image_feats)
            self.image_adj = build_knn_neighbourhood(self.image_adj, topk=self.args.topk)

            self.text_adj = build_sim(text_feats)
            self.text_adj = build_knn_neighbourhood(self.text_adj, topk=self.args.topk)   

            
            learned_adj = weight[0] * self.image_adj + weight[1] * self.text_adj
            learned_adj = compute_normalized_laplacian(learned_adj)
            original_adj = weight[0] * self.image_original_adj + weight[1] * self.text_original_adj
            self.item_adj = (1 - self.args.lambda_coeff) * learned_adj + self.args.lambda_coeff * original_adj
        else:
            self.item_adj = self.item_adj.detach()

        h = self.item_id_embedding.weight
        for i in range(self.args.n_layers):
            h = torch.mm(self.item_adj, h)

        if self.args.cf_model == 'ngcf':
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(self.n_ui_layers):
                side_embeddings = torch.sparse.mm(adj, ego_embeddings)
                sum_embeddings = F.leaky_relu(self.GC_Linear_list[i](side_embeddings))
                bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
                bi_embeddings = F.leaky_relu(self.Bi_Linear_list[i](bi_embeddings))
                ego_embeddings = sum_embeddings + bi_embeddings
                ego_embeddings = self.dropout_list[i](ego_embeddings)

                norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
                all_embeddings += [norm_embeddings]

            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)            
            u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
            i_g_embeddings = i_g_embeddings + F.normalize(h, p=2, dim=1)
            return u_g_embeddings, i_g_embeddings
        elif self.args.cf_model == 'lightgcn':
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(self.n_ui_layers):
                side_embeddings = torch.sparse.mm(adj, ego_embeddings)
                ego_embeddings = side_embeddings
                all_embeddings += [ego_embeddings]
            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
            u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
            i_g_embeddings = i_g_embeddings + F.normalize(h, p=2, dim=1)
            return u_g_embeddings, i_g_embeddings
        elif self.args.cf_model == 'mf':
                return self.user_embedding.weight, self.item_id_embedding.weight + F.normalize(h, p=2, dim=1)


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, n_layers, dropout=0.0, cat_rate=1.0):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.dropout = nn.Dropout(p=dropout)
        self.cat_rate = cat_rate  # e.g., self.args.model_cat_rate
        
        # User and Item ID embeddings
        self.user_id_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Initialization
        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

    def forward(self, ui_graph, iu_graph):
        """
        Args:
            ui_graph: sparse tensor of shape (n_users, n_items)
            iu_graph: sparse tensor of shape (n_items, n_users)
        
        Returns:
            u_embeddings: (n_users, embedding_dim)
            i_embeddings: (n_items, embedding_dim)
        """
        u_embeddings = self.user_id_embedding.weight
        i_embeddings = self.item_id_embedding.weight

        u_emb_list = [u_embeddings]
        i_emb_list = [i_embeddings]

        for l in range(self.n_layers):
            if l == (self.n_layers - 1):
                u_embeddings = F.softmax(torch.matmul(ui_graph, i_embeddings), dim=-1)
                i_embeddings = F.softmax(torch.matmul(iu_graph, u_embeddings), dim=-1)
            else:
                u_embeddings = torch.matmul(ui_graph, i_embeddings)
                i_embeddings = torch.matmul(iu_graph, u_embeddings)

            u_emb_list.append(u_embeddings)
            i_emb_list.append(i_embeddings)

        # 평균 aggregation
        final_user_emb = torch.mean(torch.stack(u_emb_list), dim=0)
        final_item_emb = torch.mean(torch.stack(i_emb_list), dim=0)

        return final_user_emb, final_item_emb

    def get_topk_recommendation(self, user_emb, item_emb, top_k=10):
        """
        Args:
            user_emb: (n_users, embedding_dim)
            item_emb: (n_items, embedding_dim)
            top_k: number of top items to recommend

        Returns:
            indices: tensor of shape (n_users, top_k), each row contains item indices
        """
        scores = torch.matmul(user_emb, item_emb.t())  # (n_users, n_items)
        topk_scores, topk_indices = torch.topk(scores, k=top_k, dim=-1)
        return topk_indices
