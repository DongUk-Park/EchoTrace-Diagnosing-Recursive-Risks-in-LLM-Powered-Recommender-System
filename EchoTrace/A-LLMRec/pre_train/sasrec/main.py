import os
import time
import torch

import torch
import argparse

from pre_train.sasrec.model import SASRec
#from pre_train.sasrec.data_preprocess import *
from pre_train.sasrec.data_preprocess_ml_1m import *
from pre_train.sasrec.utils import *

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np  

class _UserSeqDataset(Dataset):
    def __init__(self, user_train: dict, maxlen: int):
        self.users = sorted(user_train.keys())
        self.user_train = user_train
        self.maxlen = maxlen
    def __len__(self):
        return len(self.users)
    def __getitem__(self, idx):
        u = self.users[idx]
        seq = np.zeros([self.maxlen], dtype=np.int64)
        si = self.maxlen - 1
        for i in reversed(self.user_train[u]):
            if si < 0: break
            seq[si] = i
            si -= 1
        return u, seq

@torch.no_grad()
def _save_part_embeddings(model, user_train, itemnum, maxlen, save_dir, part_id, batch_size=1024, device='cpu'):
    os.makedirs(save_dir, exist_ok=True)

    # 1) Item embeddings [1..N] 저장 (0은 padding)
    item_emb = model.item_emb.weight.detach().cpu().numpy()  # (N+1, d)
    np.save(os.path.join(save_dir, f"item_emb_part_step{part_id:02d}.npy"), item_emb[1:])

    # 2) User embeddings: 마지막 non-zero 위치의 hidden (= forward(mode='log_only'))
    ds = _UserSeqDataset(user_train, maxlen)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    user_ids, user_vecs = [], []
    model.eval()
    for u_batch, seq_batch in dl:
        # seq_batch: List[List[int]] → (B, L) numpy
        seq_batch = np.stack(seq_batch, axis=0)
        with torch.no_grad():
            uvec = model(user_ids=None, log_seqs=seq_batch, pos_seqs=None, neg_seqs=None, mode='log_only')
        user_ids.extend(int(u) for u in u_batch)
        user_vecs.append(uvec.detach().cpu().numpy())

    user_vecs = np.vstack(user_vecs).astype(np.float32)  # (num_users, d)

    # 필요시 정렬(재현성/일관된 인덱싱)
    # 정렬하지 않으려면 아래 3줄을 지우세요.
    order = np.argsort(np.asarray(user_ids, dtype=np.int64))
    user_ids = np.asarray(user_ids, dtype=np.int64)[order]
    user_vecs = user_vecs[order]

    # 👉 npy로 각각 저장
    np.save(os.path.join(save_dir, f"user_ids_part_step{part_id:02d}.npy"), user_ids)   # (num_users,)
    np.save(os.path.join(save_dir, f"user_emb_part_step{part_id:02d}.npy"), user_vecs)  # (num_users, d)


def sasrec_main(dataset, path=None, item_num=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ml-1m')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_epochs', default=50, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--inference_only', default=False, action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)

    args = parser.parse_args()
    args.device = 'cuda:0'
    args.dataset = dataset
    # global dataset
    #create_name_dict_from_OMDB(data_dir='./A-LLMRec/data/ml-100k', output_dir='./A-LLMRec/data/ml-100k',)
    dataset = data_partition(args.dataset, path)

    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    itemnum = item_num if item_num is not None else itemnum
    print('user num:', usernum, 'item num:', itemnum)
    
    num_batch = len(user_train) // args.batch_size
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    
    # dataloader
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)       
    # model init
    model = SASRec(usernum, itemnum, args).to(args.device)
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass
    
    model.train()
    
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            kwargs, checkpoint = torch.load(args.state_dict_path, map_location=torch.device(args.device))
            kwargs['args'].device = args.device
            model = SASRec(**kwargs).to(args.device)
            model.load_state_dict(checkpoint)
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()
    
    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))
    
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    T = 0.0
    t0 = time.time()
    
    for epoch in tqdm(range(epoch_start_idx, args.num_epochs + 1)):
        if args.inference_only: break
        for step in range(num_batch):
            u, seq, pos, neg = sampler.next_batch()
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)

            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            if step % 100 == 0:
                print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs
    
        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            print('\n')
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                    % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

            print(str(t_valid) + ' ' + str(t_test) + '\n')
            t0 = time.time()
            model.train()
    
        if epoch == args.num_epochs:
            folder = args.dataset
            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            os.makedirs(os.path.join(f"./A-LLMRec/pre_train/sasrec/{args.dataset}"), exist_ok=True)
            torch.save([model.kwargs, model.state_dict()], os.path.join(f"./A-LLMRec/pre_train/sasrec/{args.dataset}", fname))
            
        # ----- 파트별 임베딩 저장 (epoch별 X) : 환경변수로 제어 -----
    part_env = os.getenv("SASREC_PART_ID")  # 예: "0", "1", ...
    save_dir = os.getenv("SASREC_EMB_DIR", f"./data/{args.dataset}/A-LLMRec_format/u_i_embedding_in_sasrec")
    if part_env is not None:
        try:
            part_id = int(part_env)
        except ValueError:
            part_id = 0
        _save_part_embeddings(model, user_train, itemnum, args.maxlen, save_dir, part_id, device=args.device)

    sampler.close()
    print("Done")
if __name__ == '__main__':
    dataset = "ml-1m"
    train_txt_path = "./data/ml-1m/A-LLMRec_format/ml-1m.txt"
    item_num = 3693
    sasrec_main(dataset, train_txt_path, item_num)
