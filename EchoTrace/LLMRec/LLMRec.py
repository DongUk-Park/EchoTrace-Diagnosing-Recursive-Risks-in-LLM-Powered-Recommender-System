from datetime import datetime
import math, json
import os
import random
import sys
from time import time
from tqdm import tqdm
import pickle
import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.sparse as sparse
from torch import autograd
import random

import copy
import itertools
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from utility.parser import parse_args
import Models
from Models import MM_Model, Decoder  
from utility.batch_test import *
from utility.logging import Logger
from utility.norm import build_sim, build_knn_normalized_graph




args = parse_args()

class Trainer(object):
    def __init__(self, data_config, file_path):
        self.task_name = "%s_%s_%s" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), args.dataset, args.cf_model,)
        self.logger = Logger(filename=self.task_name, is_debug=args.debug)
        self.logger.logging("PID: %d" % os.getpid())
        self.logger.logging(str(args))

        self.mess_dropout = eval(args.mess_dropout)
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.weight_size)
        self.n_layers = len(self.weight_size)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        
        self.image_feats = np.load(os.path.join(file_path, "image_feat.npy"))
        
        self.text_feats = np.load(os.path.join(file_path, "text_feat.npy"))
        print("Loading text_feat from:", os.path.join(file_path, "text_feat.npy"))
        # self.image_feats = np.load(args.data_path + '{}/image_feat.npy'.format(args.dataset))
        # self.text_feats = np.load(args.data_path + '{}/text_feat.npy'.format(args.dataset))
        self.image_feat_dim = self.image_feats.shape[-1]
        self.text_feat_dim = self.text_feats.shape[-1]

        self.ui_graph = self.ui_graph_raw = pickle.load(open(os.path.join(file_path, 'train_mat'), 'rb'))
        # get user embedding
        self.user_init_embedding = pickle.load(open(os.path.join(file_path, 'augmented_user_init_embedding_final'), 'rb'))

        # get separate embedding matrix
        self.item_attribute_embedding = pickle.load(open(os.path.join(file_path, 'augmented_total_embed_dict'), 'rb')) # i_attr step5

        self.image_ui_index = {'x':[], 'y':[]}
        self.text_ui_index = {'x':[], 'y':[]}

        self.n_users = self.ui_graph.shape[0]
        self.n_items = self.ui_graph.shape[1]
        self.iu_graph = self.ui_graph.T
  
        self.ui_graph = self.csr_norm(self.ui_graph, mean_flag=True)
        self.iu_graph = self.csr_norm(self.iu_graph, mean_flag=True)
        self.ui_graph = self.matrix_to_tensor(self.ui_graph)
        self.iu_graph = self.matrix_to_tensor(self.iu_graph)
        self.image_ui_graph = self.text_ui_graph = self.ui_graph
        self.image_iu_graph = self.text_iu_graph = self.iu_graph

        self.model_mm = MM_Model(self.n_users, self.n_items, self.emb_dim, self.weight_size, self.mess_dropout, self.image_feats, self.text_feats, self.user_init_embedding, self.item_attribute_embedding)      
        self.model_mm = self.model_mm.cuda()  
        self.decoder = Decoder(self.user_init_embedding.shape[1]).cuda()


        self.optimizer = optim.AdamW(
        [
            {'params':self.model_mm.parameters()},      
        ]
            , lr=self.lr)  
        
        self.de_optimizer = optim.AdamW(
        [
            {'params':self.decoder.parameters()},      
        ]
            , lr=args.de_lr)  

    def csr_norm(self, csr_mat, mean_flag=False):
        rowsum = np.array(csr_mat.sum(1))
        rowsum = np.power(rowsum+1e-8, -0.5).flatten()
        rowsum[np.isinf(rowsum)] = 0.
        rowsum_diag = sp.diags(rowsum)
        colsum = np.array(csr_mat.sum(0))
        colsum = np.power(colsum+1e-8, -0.5).flatten()
        colsum[np.isinf(colsum)] = 0.
        colsum_diag = sp.diags(colsum)
        if mean_flag == False:
            return rowsum_diag*csr_mat*colsum_diag
        else:
            return rowsum_diag*csr_mat

    def matrix_to_tensor(self, cur_matrix):
        if type(cur_matrix) != sp.coo_matrix:
            cur_matrix = cur_matrix.tocoo()  #
        indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))  #
        values = torch.from_numpy(cur_matrix.data)  #
        shape = torch.Size(cur_matrix.shape)
        return torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).cuda()  #

    def innerProduct(self, u_pos, i_pos, u_neg, j_neg):  
        pred_i = torch.sum(torch.mul(u_pos,i_pos), dim=-1)
        pred_j = torch.sum(torch.mul(u_neg,j_neg), dim=-1)
        return pred_i, pred_j

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)
            
    def sim(self, z1, z2):
        z1 = F.normalize(z1)  
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def feat_reg_loss_calculation(self, g_item_image, g_item_text, g_user_image, g_user_text):
        feat_reg = 1./2*(g_item_image**2).sum() + 1./2*(g_item_text**2).sum() \
            + 1./2*(g_user_image**2).sum() + 1./2*(g_user_text**2).sum()        
        feat_reg = feat_reg / self.n_items
        feat_emb_loss = args.feat_reg_decay * feat_reg
        return feat_emb_loss

    def prune_loss(self, pred, drop_rate):
        ind_sorted = np.argsort(pred.cpu().data).cuda()
        loss_sorted = pred[ind_sorted]
        remember_rate = 1 - drop_rate
        num_remember = int(remember_rate * len(loss_sorted))
        ind_update = ind_sorted[:num_remember]
        loss_update = pred[ind_update]
        return loss_update.mean()

    def mse_criterion(self, x, y, alpha=3):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        tmp_loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
        tmp_loss = tmp_loss.mean()
        loss = F.mse_loss(x, y)
        return loss

    def sce_criterion(self, x, y, alpha=1):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        loss = (1-(x*y).sum(dim=-1)).pow_(alpha)
        loss = loss.mean() 
        return loss

    def test(self, users_to_test, is_val):
        self.model_mm.eval()
        with torch.no_grad():
            ua_embeddings, ia_embeddings, *rest = self.model_mm(self.ui_graph, self.iu_graph, self.image_ui_graph, self.image_iu_graph, self.text_ui_graph, self.text_iu_graph)
        result = test_torch(ua_embeddings, ia_embeddings, users_to_test, is_val)

        return result
    
    # def get_best_candidates(self):
    #     self.model_mm.eval()
    #     with torch.no_grad():
    #         ua_embeddings, ia_embeddings, *rest = self.model_mm(self.ui_graph, self.iu_graph, self.image_ui_graph, self.image_iu_graph, self.text_ui_graph, self.text_iu_graph)
    #         # ✅ top-K candidate 반환 추가
    #         top_k = eval(args.Ks)[-1]  # 가장 작은 K 사용
    #         scores = torch.matmul(ua_embeddings, ia_embeddings.T)
    #         _, candidate_indices = torch.topk(scores, k=top_k, dim=-1) # 반환: scores, candidates
    #     return candidate_indices, ua_embeddings, ia_embeddings

    def get_best_candidates(self):
        self.model_mm.eval()
        with torch.no_grad():
            ua_embeddings, ia_embeddings, *rest = self.model_mm(
                self.ui_graph,
                self.iu_graph,
                self.image_ui_graph,
                self.image_iu_graph,
                self.text_ui_graph,
                self.text_iu_graph,
            )

            top_k = eval(args.Ks)[-1]
            top_k = min(top_k, self.n_items)

            scores = torch.matmul(ua_embeddings, ia_embeddings.T)

            # Mask items already observed in the current train matrix.
            train_coo = self.ui_graph_raw.tocoo()
            seen_users = torch.as_tensor(train_coo.row, dtype=torch.long, device=scores.device)
            seen_items = torch.as_tensor(train_coo.col, dtype=torch.long, device=scores.device)

            valid_mask = (seen_users < scores.shape[0]) & (seen_items < scores.shape[1])
            scores[seen_users[valid_mask], seen_items[valid_mask]] = -torch.inf

            _, candidate_indices = torch.topk(scores, k=top_k, dim=-1)

        return candidate_indices, ua_embeddings, ia_embeddings


    def train(self, data_generator):
        now_time = datetime.now()
        run_time = datetime.strftime(now_time,'%Y_%m_%d__%H_%M_%S')

        training_time_list = []
        stopping_step = 0

        n_batch = data_generator.n_train // args.batch_size + 1
        best_recall = 0
        best_candidates = None
        
        #for epoch in range(1):
        for epoch in range(args.epoch):
            t1 = time()
            loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
            contrastive_loss = 0.
            n_batch = data_generator.n_train // args.batch_size + 1
            sample_time = 0.
            build_item_graph = True

            self.gene_u, self.gene_real, self.gene_fake = None, None, {}
            self.topk_p_dict, self.topk_id_dict = {}, {}

            for idx in tqdm(range(n_batch)):
                self.model_mm.train()
                sample_t1 = time()
                users, pos_items, neg_items = data_generator.sample()

                # augment samples 
                augmented_sample_dict = pickle.load(open(os.path.join(args.data_path, args.dataset, 'augmented_sample_dict'), 'rb'))
                
                # 딕셔너리 키 존재 여부 확인
                users_aug = random.sample(users, int(len(users)*args.aug_sample_rate))

                valid_users_aug = []
                valid_pos_items_aug = []
                valid_neg_items_aug = []

                for user in users_aug:
                    # 1. 딕셔너리에 user 키가 존재하는지 먼저 확인 (KeyError 방지)
                    if user in augmented_sample_dict:
                        pos_item = augmented_sample_dict[user][0]
                        neg_item = augmented_sample_dict[user][1]
                        
                        # 2. item 인덱스가 유효 범위 내에 있는지 확인
                        if pos_item < self.n_items and neg_item < self.n_items:
                            valid_users_aug.append(user)
                            valid_pos_items_aug.append(pos_item)
                            valid_neg_items_aug.append(neg_item)

                # 검증된 리스트로 덮어쓰기
                users_aug = valid_users_aug
                pos_items_aug = valid_pos_items_aug
                neg_items_aug = valid_neg_items_aug

                self.new_batch_size = len(users_aug)
                users += users_aug
                pos_items += pos_items_aug
                neg_items += neg_items_aug


                sample_time += time() - sample_t1       
                user_presentation_h, item_presentation_h, image_i_feat, text_i_feat, image_u_feat, text_u_feat \
                                , user_prof_feat_pre, item_prof_feat_pre, user_prof_feat, item_prof_feat, user_att_feats, item_att_feats, i_mask_nodes, u_mask_nodes \
                        = self.model_mm(self.ui_graph, self.iu_graph, self.image_ui_graph, self.image_iu_graph, self.text_ui_graph, self.text_iu_graph)
                
                u_bpr_emb = user_presentation_h[users]
                i_bpr_pos_emb = item_presentation_h[pos_items]
                i_bpr_neg_emb = item_presentation_h[neg_items]
                batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_bpr_emb, i_bpr_pos_emb, i_bpr_neg_emb)
       
                # modal feat
                image_u_bpr_emb = image_u_feat[users]
                image_i_bpr_pos_emb = image_i_feat[pos_items]
                image_i_bpr_neg_emb = image_i_feat[neg_items]
                image_batch_mf_loss, image_batch_emb_loss, image_batch_reg_loss = self.bpr_loss(image_u_bpr_emb, image_i_bpr_pos_emb, image_i_bpr_neg_emb)
                text_u_bpr_emb = text_u_feat[users]
                text_i_bpr_pos_emb = text_i_feat[pos_items]
                text_i_bpr_neg_emb = text_i_feat[neg_items]
                text_batch_mf_loss, text_batch_emb_loss, text_batch_reg_loss = self.bpr_loss(text_u_bpr_emb, text_i_bpr_pos_emb, text_i_bpr_neg_emb)
                mm_mf_loss = image_batch_mf_loss + text_batch_mf_loss

                batch_mf_loss_aug = 0 
                for index,value in enumerate(item_att_feats):  # 
                    u_g_embeddings_aug = user_prof_feat[users]
                    pos_i_g_embeddings_aug = item_att_feats[value][pos_items]
                    neg_i_g_embeddings_aug = item_att_feats[value][neg_items]
                    tmp_batch_mf_loss_aug, batch_emb_loss_aug, batch_reg_loss_aug = self.bpr_loss(u_g_embeddings_aug, pos_i_g_embeddings_aug, neg_i_g_embeddings_aug)
                    batch_mf_loss_aug += tmp_batch_mf_loss_aug

                feat_emb_loss = self.feat_reg_loss_calculation(image_i_feat, text_i_feat, image_u_feat, text_u_feat)

                att_re_loss = 0
                if args.mask:
                    input_i = {} 
                    for index,value in enumerate(item_att_feats.keys()):  
                        input_i[value] = item_att_feats[value][i_mask_nodes]
                    decoded_u, decoded_i = self.decoder(torch.tensor(user_prof_feat[u_mask_nodes]), input_i)
                    if args.feat_loss_type=='mse':
                        att_re_loss += self.mse_criterion(decoded_u, torch.tensor(self.user_init_embedding[u_mask_nodes]).cuda(), alpha=args.alpha_l)
                        for index,value in enumerate(item_att_feats.keys()):  
                            att_re_loss += self.mse_criterion(decoded_i[index], torch.tensor(self.item_attribute_embedding[value][i_mask_nodes]).cuda(), alpha=args.alpha_l)
                    elif args.feat_loss_type=='sce':
                        att_re_loss += self.sce_criterion(decoded_u, torch.tensor(self.user_init_embedding[u_mask_nodes]).cuda(), alpha=args.alpha_l) 
                        for index,value in enumerate(item_att_feats.keys()):  
                            att_re_loss += self.sce_criterion(decoded_i[index], torch.tensor(self.item_attribute_embedding[value][i_mask_nodes]).cuda(), alpha=args.alpha_l)

                batch_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss + feat_emb_loss + args.aug_mf_rate*batch_mf_loss_aug + args.mm_mf_rate*mm_mf_loss + args.att_re_rate*att_re_loss
                nn.utils.clip_grad_norm_(self.model_mm.parameters(), max_norm=1.0)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      #+ ssl_loss2 #+ batch_contrastive_loss
                self.optimizer.zero_grad()  
                batch_loss.backward(retain_graph=False)
                
                self.optimizer.step()

                loss += float(batch_loss)
                mf_loss += float(batch_mf_loss)
                emb_loss += float(batch_emb_loss)
                reg_loss += float(batch_reg_loss)
    
            del user_presentation_h, item_presentation_h, u_bpr_emb, i_bpr_neg_emb, i_bpr_pos_emb

            if math.isnan(loss) == True:
                self.logger.logging('ERROR: loss is nan.')
                sys.exit()

            if (epoch + 1) % args.verbose != 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f  + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss, reg_loss, contrastive_loss)
                training_time_list.append(time() - t1)
                self.logger.logging(perf_str)

        best_candidates, ua_embeddings, ia_embeddings = self.get_best_candidates()

        return best_candidates, ua_embeddings, ia_embeddings

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1./(2*(users**2).sum()+1e-8) + 1./(2*(pos_items**2).sum()+1e-8) + 1./(2*(neg_items**2).sum()+1e-8)        
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores+1e-8)
        mf_loss = - self.prune_loss(maxi, args.prune_loss_drop_rate)

        emb_loss = self.decay * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)  

def hyper_param_test(data_generator):
    param_grid = {
        'batch_size': [128, 256, 512],
        'lr': [1e-3, 5e-4],
        'de_lr': [2e-4, 4e-4],
        'weight_decay': [1e-4, 1e-5, 1e-6],
        'embed_size': [64],
        'layers': [1, 2],
        'drop_rate': [0.0, 0.1, 0.2]
    }

    # 모든 조합 구하기
    keys, values = zip(*param_grid.items())
    combinations = list(itertools.product(*values))
    csv_path = "./LLMRec/LLMRec_c/movielens/ml100k_llmrec_format/output/LightGCN_grid_search_results.csv"
    results = []
    for i, combo in enumerate(combinations):
        print(f"\n🚀 Starting experiment {i+1}/{len(combinations)}")
        param_dict = dict(zip(keys, combo))
        print("▶ Params:", param_dict)

        # 파라미터 적용
        for k, v in param_dict.items():
            setattr(args, k, v)

        config = {
            'n_users': data_generator.n_users,
            'n_items': data_generator.n_items
        }

        trainer = Trainer(data_config=config, file_path=args.data_path + args.dataset + '/')
        best_recall, run_time, best_candidates = trainer.train()

        # 한 줄씩 저장
        result_entry = {**param_dict, "best_recall@20": best_recall, "time": run_time}
        df = pd.DataFrame([result_entry])
        df.to_csv(csv_path, mode='a', index=False, header=False)

        print(f"📌 Saved result for experiment {i+1} to CSV.")
    # CSV로 저장
    result_df = pd.DataFrame(results)
    result_df.to_csv("grid_search_results.csv", index=False)
    print("\n📁 All experiment results saved to grid_search_results.csv")


def main(data_path, folder):
    args.data_path = data_path
    args.dataset = folder
    Models.args.data_path = data_path
    Models.args.dataset = folder
    
    data_generator = Data(path=os.path.join(args.data_path, args.dataset), batch_size=args.batch_size)
    set_seed(args.seed)
    #hyper_param_test(data_generator)
    
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    print("LLMRec_Training: n_users:", config['n_users'], "n_items:", config['n_items'])
    trainer = Trainer(data_config=config, file_path=os.path.join(args.data_path, args.dataset) + '/')
    best_candidates, ua_embeddings, ia_embeddings = trainer.train(data_generator)
    return best_candidates, ua_embeddings, ia_embeddings

if __name__ == '__main__':
    best_candidates, ua_embeddings, ia_embeddings = main()
    # best_candidates를 CPU로 옮기고 numpy 변환
    candidates = best_candidates.cpu().numpy().tolist()

    # user별로 dict 구성
    user2cands = {str(u): cands for u, cands in enumerate(candidates)}

    # 저장 경로 설정
    save_path = "./LLMRec/LLMRec_c/movielens/ml100k_llmrec_format/best_candidates_for_feedback_loop.json"

    with open(save_path, "w") as f:
        json.dump(user2cands, f, indent=2)

    print(f"✅ user별 best_candidates 저장 완료: {save_path}")
