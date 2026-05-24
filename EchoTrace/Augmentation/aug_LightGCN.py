#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from collections import defaultdict, Counter
from functools import lru_cache
from typing import Tuple, Dict, Any, Optional, List

import os, json, gzip, pickle, re, time, math
import random
import requests

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

# =====================
# 경로/설정 (Books)
# =====================
BOOKS_DIR   = Path("./data/books")
TRAIN_PATH  = BOOKS_DIR / "train_raw.txt"         # u \t i \t YYYY.MM.DD (또는 unix)
LABEL_PATH  = BOOKS_DIR / "label.txt"         # u \t i \t YYYY.MM.DD (또는 unix)
U_ITEM_PATH = BOOKS_DIR / "item_meta_2017_kcore5_user_item.json"       # JSONL: {"item_id": new_id, "asin": "...", "title": "...", ...}

# 피드백 루프 결과 저장 경로(원하면 변경)
PRED_OUT = "./Augmentation/data/feedback_loop/books/predict_label_for_feedback_loop.json"

# =====================
# LightGCN
# =====================
class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_layers, dropout=0.0):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, adj: torch.Tensor):
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        all_emb  = torch.cat([user_emb, item_emb], dim=0)  # [U+I, D]

        assert adj.is_sparse, "adj must be a torch.sparse tensor."
        embs = [all_emb]
        for _ in range(self.num_layers):
            all_emb = torch.sparse.mm(adj, all_emb)
            if self.dropout > 0:
                all_emb = F.dropout(all_emb, p=self.dropout, training=self.training)
            embs.append(all_emb)

        final_emb = torch.mean(torch.stack(embs, dim=1), dim=1)
        user_final, item_final = final_emb[:self.num_users], final_emb[self.num_users:]
        return user_final, item_final

# =====================
# Books 데이터 로더
# =====================
def _safe_int(x: str) -> Optional[int]:
    try:
        return int(float(x))
    except Exception:
        return None

def load_interactions_file(path: Path) -> Dict[int, List[int]]:
    """
    파일 형식: user \t item \t (date or unix)
    - 날짜 문자열은 무시, 앞의 user/item만 읽어 사용자별 순서 보존 리스트를 만든다.
    """
    user2items: Dict[int, List[int]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            parts = line.split("\t")
            if len(parts) < 2: continue
            u = _safe_int(parts[0]); i = _safe_int(parts[1])
            if u is None or i is None: continue
            user2items[u].append(i)
    return user2items

def detect_num_items_from_meta(u_item_path: Path) -> int:
    """
    u_item.json(JSONL)의 item_id 최댓값 + 1을 반환.
    파일이 없으면 0 반환.
    """
    if not u_item_path.exists():
        return 0
    max_id = -1
    with open(u_item_path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                iid = int(obj.get("item_id"))
                if iid > max_id: max_id = iid
            except Exception:
                continue
    return max_id + 1 if max_id >= 0 else 0

def get_books_data() -> Tuple[Dict[int, List[int]], Dict[int, List[int]], set, set, int, int]:
    """
    Books용 상호작용 로딩:
    - train.txt를 '전체 상호작용'으로 보고 사용자별 시퀀스를 구성
    - 이후 사용자별 70/30 split (순서 보존)으로 train/test 분리
    - cold item은 train(70%)에 등장하지 않은 아이템
    - 반환: train_dict_train, test_dict, warm_items, cold_items, num_users, num_items_total
    """
    full = load_interactions_file(TRAIN_PATH)  # 시간 기준 앞절반이 이미 반영된 파일
    # 사용자별 순서 보존 70/30
    train_d, test_d = split_train_test_by_ratio(full, train_ratio=0.7)

    # num_users/num_items_total 추정
    max_u = max(full.keys()) if full else 0
    num_users = max_u + 1
    # 메타에서 아이템 수 우선 탐지, 안 되면 상호작용 기반으로
    num_items_total = detect_num_items_from_meta(U_ITEM_PATH)
    if num_items_total <= 0:
        max_item = -1
        for items in full.values():
            if items:
                max_item = max(max_item, max(items))
        num_items_total = max_item + 1 if max_item >= 0 else 0

    # cold/warm (train 70% 기준)
    warm = set()
    for items in train_d.values():
        warm.update(items)
    all_items = set(range(num_items_total))
    cold = all_items - warm

    return train_d, test_d, warm, cold, num_users, num_items_total

# =====================
# 그래프 유틸
# =====================
def build_norm_adj(train_dict: Dict[int, List[int]], num_users: int, num_items: int) -> torch.Tensor:
    rows, cols = [], []
    for u, items in train_dict.items():
        for i in set(items):
            rows += [u, num_users + i]
            cols += [num_users + i, u]

    N = num_users + num_items
    if not rows:
        indices = torch.empty((2, 0), dtype=torch.long)
        values  = torch.empty((0,), dtype=torch.float32)
        return torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()

    row = torch.tensor(rows, dtype=torch.long)
    col = torch.tensor(cols, dtype=torch.long)
    deg = torch.bincount(row, minlength=N).to(torch.float32)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    values = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    indices = torch.stack([row, col], dim=0)
    return torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()

# =====================
# 메타 로딩 (Books)
# =====================
@lru_cache(maxsize=1)
def _load_books_meta() -> Dict[int, Dict[str, Any]]:
    """
    u_item.json(JSONL) → item_id -> meta(dict)
    description이 없으므로 간단한 설명을 합성해 제공.
    """
    meta = {}
    if not U_ITEM_PATH.exists():
        return meta
    with open(U_ITEM_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                iid = int(obj.get("item_id"))
                asin = (obj.get("asin") or "").strip()
                title = (obj.get("title") or "").strip()
                brand = (obj.get("brand") or "").strip()
                # category가 리스트일 수 있으므로 문자열로 변환
                category = obj.get("category") 
                if isinstance(category, list):
                    category = "|".join([str(c).strip() for c in category])
                else:
                    category = str(category or "").strip()
                # description이 리스트일 수 있으므로 문자열로 변환
                desc = obj.get("description")
                if isinstance(desc, list):
                    desc = " ".join([str(d).strip() for d in desc])
                else:
                    desc = str(desc or "").strip()
                meta[iid] = {
                    "title": title,
                    "desc": f"[brand:{brand}], [category:{category}], [description:{desc}]",
                    "asin": asin
                }
            except Exception:
                continue
    return meta

def get_item_info(item_id: int) -> Tuple[str, str]:
    m = _load_books_meta()
    info = m.get(int(item_id))
    if info is None:
        return f"Item-{item_id}", ""
    return info.get("title") or f"Item-{item_id}", info.get("desc") or ""

# =====================
# (선택) LLM API
# =====================
def llm_api_call(prompt: str,
                 *,
                 model_type: Optional[str] = None,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 timeout: int = 30,
                 max_retries: int = 2) -> str:
    model = model_type or os.getenv("OPENAI_MODEL", "gpt-4o")
    key   = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")

    url_root = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
    url = url_root.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    messages = [
        {"role": "system", "content": "You are a strict judge that outputs only valid JSON."},
        {"role": "user", "content": prompt + '\nReturn a JSON object like {"item_id": <number>} and nothing else.'},
    ]
    payload = {
        "model": model, "messages": messages, "temperature": 0.2, "max_tokens": 16, "stream": False,
        "response_format": {"type": "json_object"},
    }

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 400 and "response_format" in resp.text:
                payload.pop("response_format", None)
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            try:
                obj = json.loads(content)
                return str(int(obj.get("item_id")))
            except Exception:
                m = re.search(r"-?\d+", content)
                if m:
                    return m.group(0).lstrip("+")
                raise ValueError(f"LLM 응답 파싱 실패: {content[:200]}")
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(1.2 * (attempt + 1))
                continue
            break
    assert last_err is not None
    raise last_err

# =====================
# 증강 (옵션)
# =====================
def augment_data(train_dict: Dict[int, List[int]], cold_items: set) -> List[Tuple[int,int,int]]:
    users = list(train_dict.keys())
    # users의 20%만 랜덤추출하여 증강 진행, seed=42 고정
    random.seed(42)
    users = random.sample(users, k=max(1, len(users)//10))
    
    cold_list = list(cold_items)
    aug_triplets = []
    if len(cold_list) < 2:
        return aug_triplets
    
    for idx, u in enumerate(tqdm(users, desc="Augmenting data")):
        hist = train_dict[u]
        a_idx, b_idx = torch.randint(0, len(cold_list), (2,)).tolist()
        if a_idx == b_idx: continue
        a, b = cold_list[a_idx], cold_list[b_idx]

        # ---- 프롬프트 구성
        def _hist_str():
            s = []
            for h in hist[-10:]:
                th, dh = get_item_info(h)
                s.append(f"[id={h}, title={th}, desc={dh}]")
            return "\n".join(s)

        tA, dA = get_item_info(a)
        tB, dB = get_item_info(b)
        prompt = (
            f"The user consumed (ordered):\n{_hist_str()}\n\n"
            f"Predict the next preference: A or B.\n"
            f"A: [id={a}, title={tA}, desc={dA}]\n"
            f"B: [id={b}, title={tB}, desc={dB}]\n"
            f'Return JSON like {{"item_id": <number>}}'
        )
        # 실제 LLM 호출
        try:
            choice = int(llm_api_call(prompt))
        except Exception:
            choice = a

        pos = a if choice == a else b
        neg = b if pos == a else a
        aug_triplets.append((u, pos, neg))
    return aug_triplets

# =====================
# Dataset / Loss
# =====================
class MainTrainDataset(Dataset):
    def __init__(self, train_dict, num_items, neg_k=10):
        self.samples = []
        self.neg_k = neg_k
        self.num_items = num_items
        self.user_pos = {u: set(items) for u, items in train_dict.items()}
        all_items = set(range(num_items))
        self.user_neg_pool = {u: list(all_items - pos) for u, pos in self.user_pos.items()}
        for u, items in train_dict.items():
            for i in items:
                self.samples.append((u, i))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        u, pos = self.samples[idx]
        neg_pool = self.user_neg_pool[u]
        if len(neg_pool) == 0:
            negs = [0]*self.neg_k
        else:
            negs = [neg_pool[random.randint(0, len(neg_pool)-1)] for _ in range(self.neg_k)]
        return int(u), int(pos), torch.tensor(negs, dtype=torch.long)

def sampled_softmax_loss(u_ids, pos_i_ids, neg_i_ids, user_emb, item_emb):
    u = user_emb[u_ids]
    pos = item_emb[pos_i_ids]
    neg = item_emb[neg_i_ids]          # [B, K, D]
    pos_logit = (u * pos).sum(dim=1, keepdim=True)
    neg_logits = torch.einsum('bd,bkd->bk', u, neg)
    logits = torch.cat([pos_logit, neg_logits], dim=1)
    targets = torch.zeros(u.shape[0], dtype=torch.long, device=u_ids.device)
    return F.cross_entropy(logits, targets)

def bpr_loss(u_ids, pos_i_ids, neg_i_ids, user_emb, item_emb):
    u = user_emb[u_ids]; pos = item_emb[pos_i_ids]; neg = item_emb[neg_i_ids]
    x = (u*pos).sum(dim=1) - (u*neg).sum(dim=1)
    return -F.logsigmoid(x).mean()

# =====================
# Split helper
# =====================
def split_train_test_by_ratio(user_dict, train_ratio=0.7):
    train_d, test_d = {}, {}
    for u, items in user_dict.items():
        n = len(items)
        if n == 0:
            train_d[u], test_d[u] = [], []
            continue
        train_n = max(1, int(math.floor(n * train_ratio)))
        train_d[u] = items[:train_n]
        test_d[u]  = items[train_n:]
    return train_d, test_d

# =====================
# Evaluation
# =====================
def evaluate_recall_precision_hitratio(model, adj_t, train_dict_train, test_dict, num_items_total, Ks=(10,20)):
    model.eval()
    results = {K: {"recall":0.0,"precision":0.0,"hit":0,"users":0} for K in Ks}
    Kmax = max(Ks)
    with torch.no_grad():
        user_emb, item_emb = model(adj_t)
        device = user_emb.device
        for u, test_items in test_dict.items():
            if not test_items: continue
            scores = (user_emb[u:u+1] @ item_emb.T).squeeze(0)
            # seen 마스킹
            seen = train_dict_train.get(u, [])
            if seen:
                idx = torch.tensor(seen, dtype=torch.long, device=device)
                scores.index_fill_(0, idx, -1e9)
            topk_idx = torch.topk(scores, k=min(Kmax, scores.numel())).indices.tolist()
            test_set = set(test_items)
            for K in Ks:
                predK = set(topk_idx[:K])
                hits = len(predK & test_set)
                results[K]["recall"]   += hits / len(test_set)
                results[K]["precision"] += hits / max(1, K)
                results[K]["hit"]      += 1 if hits>0 else 0
                results[K]["users"]    += 1
    final = {}
    for K in Ks:
        n = results[K]["users"]
        final[K] = {
            "recall":   results[K]["recall"]/n if n else 0.0,
            "precision":results[K]["precision"]/n if n else 0.0,
            "hitratio": results[K]["hit"]/n if n else 0.0,
            "users":    n,
        }
    return final

def evaluate_table1_grouped_recall(model, adj_t, train_dict_train, test_dict, cold_items_set, num_items_total, Ks=(5,10,20), mask_seen=True):
    model.eval()
    Kmax = max(Ks)
    groups = {'cold': {'hits': {K:0 for K in Ks}, 'count':0},
              'warm': {'hits': {K:0 for K in Ks}, 'count':0}}
    with torch.no_grad():
        user_emb, item_emb = model(adj_t)
        device = user_emb.device
        for u, test_items in test_dict.items():
            if not test_items: continue
            scores = (user_emb[u:u+1] @ item_emb.T).squeeze(0)
            if mask_seen:
                seen = train_dict_train.get(u, [])
                if seen:
                    idx = torch.tensor(seen, dtype=torch.long, device=device)
                    scores.index_fill_(0, idx, -1e9)
            topk_idx = torch.topk(scores, k=min(Kmax, scores.numel())).indices.tolist()
            topk_sets = {K:set(topk_idx[:K]) for K in Ks}
            for i_true in test_items:
                label = 'cold' if i_true in cold_items_set else 'warm'
                groups[label]['count'] += 1
                for K in Ks:
                    if i_true in topk_sets[K]:
                        groups[label]['hits'][K] += 1
    rows=[]
    for K in Ks:
        c_cnt=groups['cold']['count']; w_cnt=groups['warm']['count']
        c_r=(groups['cold']['hits'][K]/c_cnt) if c_cnt>0 else 0.0
        w_r=(groups['warm']['hits'][K]/w_cnt) if w_cnt>0 else 0.0
        a_r=((groups['cold']['hits'][K]+groups['warm']['hits'][K])/(c_cnt+w_cnt)) if (c_cnt+w_cnt)>0 else 0.0
        rows.append({'K':K,'R@K (cold-start)':c_r,'R@K (warm-start)':w_r,'R@K (all)':a_r,'N_col':c_cnt,'N_warm':w_cnt})
    if pd is not None:
        return pd.DataFrame(rows).set_index('K'), groups
    return rows, groups

# =====================
# 학습 루틴
# =====================
def train_lightgcn_with_aug(train_dict, aug_triplets,
                            embedding_dim=64, num_layers=2, neg_k=10,
                            epochs=20, batch_size=2048, lr=1e-3, dropout=0.0,
                            num_users=0, num_items_total=0, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LightGCN(num_users=num_users, num_items=num_items_total,
                     embedding_dim=embedding_dim, num_layers=num_layers, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    adj_t = build_norm_adj(train_dict, num_users, num_items_total).to(device)

    main_ds = MainTrainDataset(train_dict, num_items=num_items_total, neg_k=neg_k)
    main_loader = DataLoader(main_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    if aug_triplets:
        aug_U = torch.tensor([t[0] for t in aug_triplets], dtype=torch.long, device=device)
        aug_P = torch.tensor([t[1] for t in aug_triplets], dtype=torch.long, device=device)
        aug_N = torch.tensor([t[2] for t in aug_triplets], dtype=torch.long, device=device)
    else:
        aug_U = aug_P = aug_N = None

    for ep in range(1, epochs+1):
        model.train()
        total_loss=total_main=total_bpr=0.0; steps=0
        for u, pos, negs in main_loader:
            u=u.to(device); pos=pos.to(device); negs=negs.to(device)
            user_emb, item_emb = model(adj_t)
            loss_main = sampled_softmax_loss(u, pos, negs, user_emb, item_emb)
            if aug_U is not None and aug_U.numel()>0:
                idx = torch.randint(0, aug_U.shape[0], (u.shape[0],), device=device)
                u_a, p_a, n_a = aug_U[idx], aug_P[idx], aug_N[idx]
                loss_bpr = bpr_loss(u_a, p_a, n_a, user_emb, item_emb)
            else:
                loss_bpr = torch.tensor(0.0, device=device)
            loss = loss_main + loss_bpr

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item(); total_main += loss_main.item()
            total_bpr  += float(loss_bpr.item()) if isinstance(loss_bpr, torch.Tensor) else 0.0
            steps += 1
        print(f"[Epoch {ep:02d}] loss={total_loss/steps:.4f} | main={total_main/steps:.4f} | aug_bpr={total_bpr/steps:.4f}")

    return model, adj_t

def save_topk_predictions_for_feedback_loop(model, adj_t, train_dict_train, num_items_total, K: int, out_path: str):
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model(adj_t)
        device = user_emb.device
        result = {}
        for u in train_dict_train.keys():
            scores = (user_emb[u:u+1] @ item_emb.T).squeeze(0)
            # seen 마스킹
            seen = train_dict_train.get(u, [])
            if seen:
                idx = torch.tensor(seen, dtype=torch.long, device=device)
                scores.index_fill_(0, idx, -1e9)
            valid = (scores > -1e9/2).nonzero(as_tuple=False).flatten()
            if valid.numel() == 0:
                result[str(u)] = []
            else:
                k_eff = min(K, valid.numel())
                topk_idx = torch.topk(scores, k=k_eff).indices.tolist()
                result[str(u)] = [int(i) for i in topk_idx]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f)
    print(f"✅ Saved Top-{K} predictions to: {out_path} (users={len(result)})")

# =====================
# 메인 실행
# =====================
if __name__ == "__main__":
    # 1) Books 데이터 적재 및 70/30 분할
    train_dict_train, test_dict, warm_items, cold_items_for_aug, num_users, num_items_total = get_books_data()
    print(f"[DATA] users={num_users}, items={num_items_total}, warm={len(warm_items)}, cold={len(cold_items_for_aug)}")

    # 2) (선택) 증강 트리플릿 생성 또는 로드
    aug_triplets = augment_data(train_dict_train, cold_items_for_aug)
    with open(BOOKS_DIR / "aug_triplets.pkl", "wb") as f: pickle.dump(aug_triplets, f)
    #with open(BOOKS_DIR / "aug_triplets.pkl", "rb") as f: aug_triplets = pickle.load(f)
    print(f"Augmented triplets: {len(aug_triplets)}")

    # 3) 학습
    model, adj_t = train_lightgcn_with_aug(
        train_dict=train_dict_train,
        aug_triplets=aug_triplets,
        embedding_dim=64,
        num_layers=2,
        neg_k=10,
        epochs=30,
        batch_size=2048,
        lr=1e-3,
        dropout=0.0,
        num_users=num_users,
        num_items_total=num_items_total,
    )

    # 4) 평가 + 피드백루프 후보 저장
    Ks = (10, 20, 50, 100)
    Kmin = min(Ks)

    save_topk_predictions_for_feedback_loop(
        model=model,
        adj_t=adj_t,
        train_dict_train=train_dict_train,
        num_items_total=num_items_total,
        K=Kmin,
        out_path=PRED_OUT,
    )

    table1_df, raw = evaluate_table1_grouped_recall(
        model=model,
        adj_t=adj_t,
        train_dict_train=train_dict_train,
        test_dict=test_dict,
        cold_items_set=cold_items_for_aug,
        num_items_total=num_items_total,
        Ks=Ks,
        mask_seen=True,
    )
    print("\n=== Table 1 (Grouped R@K) ===")
    print(table1_df)
