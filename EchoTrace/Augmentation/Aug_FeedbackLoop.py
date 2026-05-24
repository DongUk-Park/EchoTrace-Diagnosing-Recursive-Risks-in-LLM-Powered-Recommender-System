#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
import json
import time
import math
import pickle
import random
import requests
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from concurrent.futures import ThreadPoolExecutor, as_completed

# =====================
# Global config
# =====================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

API_KEY = os.environ.get("OPENAI_API_KEY")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# ---------------------
# Dataset selection
# ---------------------
CODES_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = CODES_ROOT / "data"

DATASET_NAME = os.getenv("DATASET_NAME", "yelp").lower()
# choices: amazon_books, yelp, ml-1m

def get_dataset_config(dataset_name: str) -> Dict[str, str]:
    dataset_name = dataset_name.lower()

    if dataset_name == "amazon_books":
        data_root = DATA_ROOT / "books"
        return {
            "name": "amazon_books",
            "data_root": str(data_root),
            "work_dir": str(data_root / "Augmentation_format"),
            "train_src": str(data_root / "train.txt"),
            "label_src": str(data_root / "label.txt"),
            "meta_src": str(data_root / "item_meta_2017_kcore10_user_item_split_filtered.json"),
        }

    elif dataset_name == "yelp":
        data_root = DATA_ROOT / "yelp"
        return {
            "name": "yelp",
            "data_root": str(data_root),
            "work_dir": str(data_root / "Augmentation_format"),
            "train_src": str(data_root / "train.txt"),
            "label_src": str(data_root / "label.txt"),
            "meta_src": str(data_root / "item_meta_2018_kcore5_user_item_split_filtered.json"),
        }

    elif dataset_name == "ml-1m":
        data_root = DATA_ROOT / "ml-1m"
        return {
            "name": "ml-1m",
            "data_root": str(data_root),
            "work_dir": str(data_root / "Augmentation_format"),
            "train_src": str(data_root / "train.txt"),
            "label_src": str(data_root / "label.txt"),
            "meta_src": str(data_root / "item_attribute.csv"),
        }

    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

CFG = get_dataset_config(DATASET_NAME)
DATA_ROOT_DIR = CFG["data_root"]
WORK_DIR = CFG["work_dir"]
TRAIN_SRC = CFG["train_src"]
LABEL_SRC = CFG["label_src"]
META_SRC = CFG["meta_src"]


def get_run_dir(alpha: float) -> str:
    return os.path.join(WORK_DIR, f"alpha{alpha}")

# =====================
# Utils
# =====================

def ensure_int_dict(d: Dict[Any, Any]) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for k, v in d.items():
        try:
            ki = int(k)
        except Exception:
            continue
        if isinstance(v, list):
            out[ki] = [int(x) for x in v]
        else:
            try:
                out[ki] = [int(v)]
            except Exception:
                out[ki] = []
    return out

def compute_universe(train_dict: Dict[int, List[int]], test_dict: Dict[int, List[int]]) -> Tuple[int, int]:
    user_ids = set(train_dict.keys()) | set(test_dict.keys())
    item_ids: set[int] = set()
    for d in (train_dict, test_dict):
        for items in d.values():
            item_ids.update(int(i) for i in items)
    num_users_total = (max(user_ids) + 1) if user_ids else 0
    num_items_total = (max(item_ids) + 1) if item_ids else 0
    return num_users_total, num_items_total

def setdefault_list(d: Dict[str, List[int]], k: int) -> List[int]:
    ks = str(k)
    if ks not in d:
        d[ks] = []
    return d[ks]

def to_numpy_cpu(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

def seed_worker(_):
    worker_seed = SEED
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# =====================
# Model
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
        all_emb = torch.cat([user_emb, item_emb], dim=0)

        assert adj.is_sparse, "adj must be a torch.sparse tensor (COO/CSR)."

        embs = [all_emb]
        for _ in range(self.num_layers):
            all_emb = torch.sparse.mm(adj, all_emb)
            if self.dropout > 0:
                all_emb = F.dropout(all_emb, p=self.dropout, training=self.training)
            embs.append(all_emb)

        final_emb = torch.mean(torch.stack(embs, dim=1), dim=1)
        user_final_emb, item_final_emb = final_emb[:self.num_users], final_emb[self.num_users:]
        return user_final_emb, item_final_emb

# =====================
# Data utils
# =====================

def load_interaction_txt(path: str, has_timestamp: bool = False) -> Dict[int, List[int]]:
    """
    공백/탭 구분 user_id item_id [timestamp] 형태 txt 로드
    """
    data: Dict[int, List[int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = re.split(r"[\s\t]+", line.strip())
            if len(parts) < 2:
                continue
            try:
                u = int(parts[0])
                i = int(parts[1])
            except Exception:
                continue
            if u not in data:
                data[u] = []
            data[u].append(i)
    return data

def load_label_df(path: str, dataset_name: str) -> pd.DataFrame:
    """
    label.txt를 DataFrame으로 로드.
    기대 컬럼: user_id, item_id, timestamp
    timestamp가 없으면 가짜 순번 timestamp 생성
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            parts = re.split(r"[\s\t]+", line.strip())
            if len(parts) < 2:
                continue

            try:
                user_id = int(parts[0])
                item_id = int(parts[1])
            except Exception:
                continue

            if len(parts) >= 3:
                ts_raw = parts[2]
                try:
                    timestamp = int(float(ts_raw))
                except Exception:
                    # 문자열 시간 포맷이면 pandas로 파싱 시도
                    try:
                        timestamp = int(pd.Timestamp(ts_raw).timestamp())
                    except Exception:
                        timestamp = idx
            else:
                # timestamp가 아예 없으면 순번 기반 가상 timestamp
                timestamp = idx

            rows.append((user_id, item_id, timestamp))

    df = pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp"])
    return df

def get_gt(df: pd.DataFrame):
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")

    # timestamp가 가상 순번일 경우 NaT가 생길 수 있음 -> 순번 자체로 대체
    if df["date"].isna().any():
        df["date"] = pd.RangeIndex(start=0, stop=len(df), step=1).astype(str)
    else:
        df["date"] = df["date"].dt.strftime('%Y-%m-%d')

    unique_dates = sorted(df["date"].unique())
    date2idx = {date: idx for idx, date in enumerate(unique_dates)}

    data = {}
    for user_id, group in df.groupby("user_id"):
        pairs = []
        for _, row in group.iterrows():
            item_id = int(row["item_id"])
            date_idx = date2idx[row["date"]]
            pairs.append((item_id, date_idx))
        data[int(user_id)] = pairs
    return data, date2idx

def get_initial_data(save_dir: str, dataset_name: str):
    os.makedirs(save_dir, exist_ok=True)

    print(f"[{dataset_name}] Loading raw data from:")
    print(f" Train: {TRAIN_SRC}")
    print(f" Label: {LABEL_SRC}")

    train_dict = load_interaction_txt(TRAIN_SRC)
    label_dict = load_interaction_txt(LABEL_SRC)

    label_df = load_label_df(LABEL_SRC, dataset_name)
    ground_truth, date2idx = get_gt(label_df)

    with open(os.path.join(save_dir, "train.json"), "w") as f:
        json.dump(train_dict, f)
    with open(os.path.join(save_dir, "label.json"), "w") as f:
        json.dump(label_dict, f)

    print("✅ Saved train.json, label.json to work dir.")
    return ground_truth, date2idx

def get_data(data_dir: str):
    with open(os.path.join(data_dir, "train.json"), "r") as f:
        train_raw = json.load(f)
    with open(os.path.join(data_dir, "label.json"), "r") as f:
        label_raw = json.load(f)

    train_dict = ensure_int_dict(train_raw)
    label_dict = ensure_int_dict(label_raw)

    num_users_total, num_items_total = compute_universe(train_dict, label_dict)

    warm_items: set[int] = set()
    for items in train_dict.values():
        warm_items.update(items)

    all_items = set(range(num_items_total))
    cold_items = all_items - warm_items

    print(f"Users: {num_users_total} | Items: {num_items_total} | warm: {len(warm_items)} | cold: {len(cold_items)}")
    return label_dict, train_dict, warm_items, cold_items, num_users_total, num_items_total

def build_norm_adj(train_dict: Dict[int, List[int]], num_users: int, num_items: int) -> torch.Tensor:
    rows, cols = [], []
    for u, items in train_dict.items():
        if not (0 <= u < num_users):
            continue
        for i in set(items):
            if not (0 <= i < num_items):
                continue
            u_idx = u
            v_idx = num_users + i
            rows += [u_idx, v_idx]
            cols += [v_idx, u_idx]

    N = num_users + num_items
    if len(rows) == 0:
        indices = torch.empty((2, 0), dtype=torch.long)
        values = torch.empty((0,), dtype=torch.float32)
        return torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()

    row = torch.tensor(rows, dtype=torch.long)
    col = torch.tensor(cols, dtype=torch.long)

    deg = torch.bincount(row, minlength=N).to(torch.float32)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    values = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    indices = torch.stack([row, col], dim=0)
    adj = torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()
    return adj

# =====================
# Metadata loaders
# =====================

def load_amazon_meta(path: str) -> Dict[str, Any]:
    meta_dict = {}
    if not os.path.exists(path):
        print(f"Meta file not found: {path}")
        return meta_dict

    print(f"Loading Amazon metadata from {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                key = str(item.get("item_id", item.get("asin")))
                meta_dict[key] = item
            except Exception:
                continue
    print(f"Loaded {len(meta_dict)} Amazon items.")
    return meta_dict

def load_yelp_meta(path: str) -> Dict[str, Any]:
    """
    Yelp meta format: JSONL records with item_id, business_id, name,
    address, city, state, categories, stars, review_count, etc.
    """
    if not os.path.exists(path):
        print(f"Meta file not found: {path}")
        return {}

    print(f"Loading Yelp metadata from {path} ...")
    meta_dict: Dict[str, Any] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                item_id = item.get("item_id")
                if item_id is None:
                    continue
                meta_dict[str(int(item_id))] = item

        print(f"Loaded {len(meta_dict)} Yelp items.")
        return meta_dict

    except Exception as e:
        print(f"Error loading Yelp metadata: {e}")
        return {}

def load_ml1m_meta(path: str) -> Dict[str, Any]:
    """
    item_attribute.csv 형식:
    item_id,year,title,genres
    """
    meta_dict = {}
    if not os.path.exists(path):
        print(f"Meta file not found: {path}")
        return meta_dict

    print(f"Loading MovieLens-1M metadata from {path} ...")
    df = pd.read_csv(
        path,
        header=None,
        names=["item_id", "year", "title", "genres"],
        encoding="utf-8",
    )
    for _, row in df.iterrows():
        try:
            item_id = int(row["item_id"])
        except Exception:
            continue

        meta_dict[str(item_id)] = {
            "movie_id": item_id,
            "year": None if pd.isna(row["year"]) else int(row["year"]),
            "title": "" if pd.isna(row["title"]) else str(row["title"]),
            "genres": "" if pd.isna(row["genres"]) else str(row["genres"]),
        }

    print(f"Loaded {len(meta_dict)} ML-1M items.")
    return meta_dict

@lru_cache(maxsize=1)
def _load_meta_cached(dataset_name: str) -> Dict[str, Any]:
    if dataset_name == "amazon_books":
        return load_amazon_meta(META_SRC)
    elif dataset_name == "yelp":
        return load_yelp_meta(META_SRC)
    elif dataset_name == "ml-1m":
        return load_ml1m_meta(META_SRC)
    else:
        return {}

def get_item_info(item_id: int, meta: Dict[str, Any], dataset_name: str) -> Tuple[str, str]:
    sid = str(item_id)
    item_data = meta.get(sid, {})

    try:
        if dataset_name == "amazon_books":
            title = item_data.get("title") or f"Item-{item_id}"
            desc = item_data.get("description") or "None"
            brand = item_data.get("brand") or "None"
            category = item_data.get("category") or "None"
            text = f"Brand:[{brand}], Category:[{category}], Description:[{desc}]"
            return title, text

        elif dataset_name == "yelp":
            title = item_data.get("name") or f"Business-{item_id}"
            address = item_data.get("address") or "None"
            city = item_data.get("city") or "None"
            state = item_data.get("state") or "None"
            categories = item_data.get("categories") or "None"
            stars = item_data.get("stars")
            review_count = item_data.get("review_count")
            business_id = item_data.get("business_id") or "None"

            text = (
                f"BusinessID:[{business_id}], "
                f"Location:[{address}, {city}, {state}], "
                f"Categories:[{categories}], "
                f"Stars:[{stars if stars is not None else 'None'}], "
                f"ReviewCount:[{review_count if review_count is not None else 'None'}]"
            )
            return title, text

        elif dataset_name == "ml-1m":
            title = item_data.get("title") or f"Movie-{item_id}"
            genres = item_data.get("genres") or "None"
            text = f"Genres:[{genres}]"
            return title, text

        else:
            return f"Item-{item_id}", ""

    except Exception:
        return f"Item-{item_id}", ""

# =====================
# Augmentation with LLM
# =====================

def llm_api_call(prompt: str, model_type: Optional[str] = None) -> str:
    model = model_type or DEFAULT_MODEL
    key = API_KEY
    if not key:
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }
    messages = [
        {"role": "system", "content": "You are a strict judge that outputs only valid JSON."},
        {"role": "user", "content": prompt + '\nReturn a JSON object like: {"item_id": <number>} and nothing else.'},
    ]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 16,
        "response_format": {"type": "json_object"},
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        obj = json.loads(content)
        return str(int(obj.get("item_id")))
    except Exception as e:
        if 'content' in locals():
            m = re.search(r"-?\d+", content)
            if m:
                return m.group(0)
        raise e

def build_preference_prompt(
    dataset_name: str,
    history: List[int],
    itemA: int,
    itemB: int,
    meta: Dict[str, Any]
) -> str:
    hist_list = history[-10:]
    history_str = ""
    for h in hist_list:
        th, dh = get_item_info(h, meta, dataset_name)
        history_str += f"[id={h}, title={th}]\n"

    tA, dA = get_item_info(itemA, meta, dataset_name)
    tB, dB = get_item_info(itemB, meta, dataset_name)

    if dataset_name == "amazon_books":
        domain_text = "books"
        action_text = "buy next"
    elif dataset_name == "yelp":
        domain_text = "local businesses and restaurants"
        action_text = "visit or review next"
    elif dataset_name == "ml-1m":
        domain_text = "movies"
        action_text = "watch next"
    else:
        domain_text = "items"
        action_text = "choose next"

    prompt = (
        f"The user previously interacted with the following {domain_text}:\n"
        f"{history_str}\n\n"
        f"Predict which candidate the user is more likely to {action_text}.\n"
        f"A: [id={itemA}, title={tA}, desc={dA}]\n"
        f"B: [id={itemB}, title={tB}, desc={dB}]\n"
        f"Respond with only the chosen item id."
    )
    return prompt

def call_llm(
    user: int,
    history: List[int],
    items: Tuple[int, int],
    meta=None,
    dataset_name: str = "amazon_books"
) -> int:
    itemA, itemB = items
    try:
        prompt = build_preference_prompt(dataset_name, history, itemA, itemB, meta)
        resp = llm_api_call(prompt)
        return int(str(resp).strip())
    except Exception:
        return random.choice([itemA, itemB])

# =====================
# Multi-thread Helper
# =====================

def _augment_single_user(params):
    u, hist, cold_list, pairs_per_user, meta, seed, dataset_name = params

    local_rng = random.Random(seed + u)
    local_triplets = []
    l_count_a = 0
    l_count_b = 0
    l_count_h = 0

    try:
        for _ in range(pairs_per_user):
            a, b = local_rng.sample(cold_list, 2)
            choice = call_llm(u, hist, (a, b), meta=meta, dataset_name=dataset_name)

            if choice not in (a, b):
                print(f"User {u}, item {a}, {b}: Hallucination detected. LLM response: {choice}")
                choice = local_rng.choice([a, b])
                l_count_h += 1

                if choice == a:
                    pos, neg = a, b
                else:
                    pos, neg = b, a

            elif choice == a:
                l_count_a += 1
                pos, neg = a, b

            else:
                l_count_b += 1
                pos, neg = b, a

            local_triplets.append((u, int(pos), int(neg)))

    except Exception as e:
        print(f"\nError processing user {u}: {e}")
        return [], 0, 0, 0

    return local_triplets, l_count_a, l_count_b, l_count_h

def augment_data(
    train_dict: Dict[int, List[int]],
    cold_items: set[int],
    pairs_per_user: int = 1,
    rng_seed: int = 42,
    dataset_name: str = "amazon_books"
):
    rng = random.Random(rng_seed)
    users = list(train_dict.keys())

    sample_size = max(1, len(users) // 5)
    users = rng.sample(users, sample_size)

    cold_list = list(cold_items)
    aug_triplets: List[Tuple[int, int, int]] = []

    if len(cold_list) < 2:
        return aug_triplets, (0, 0, 0)

    count_a, count_b, count_h = 0, 0, 0
    meta = _load_meta_cached(dataset_name)

    MAX_WORKERS = 8
    print(f"[{dataset_name}] Starting augmentation for {len(users)} users with {MAX_WORKERS} threads...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_user = {
            executor.submit(
                _augment_single_user,
                (u, train_dict.get(u, []), cold_list, pairs_per_user, meta, rng_seed, dataset_name)
            ): u
            for u in users
        }

        for i, future in enumerate(as_completed(future_to_user)):
            u = future_to_user[future]
            try:
                triplets, c_a, c_b, c_h = future.result()
                aug_triplets.extend(triplets)
                count_a += c_a
                count_b += c_b
                count_h += c_h

                if (i + 1) % 10 == 0:
                    print(f"Augmenting progress: {i + 1}/{len(users)} users done...", end="\r")

            except Exception as exc:
                print(f"\nUser {u} generated an exception: {exc}")

    print(f"\nAugmentation Complete. Total triplets: {len(aug_triplets)}")
    return aug_triplets, (count_a, count_b, count_h)

# =====================
# Datasets & Losses
# =====================

class MainTrainDataset(Dataset):
    def __init__(self, train_dict: Dict[int, List[int]], num_items: int, neg_k: int = 10):
        self.samples: List[Tuple[int, int]] = []
        self.neg_k = neg_k
        self.num_items = num_items
        self.user_pos = {u: set(items) for u, items in train_dict.items()}

        for u, items in train_dict.items():
            for i in items:
                if 0 <= i < num_items:
                    self.samples.append((u, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        u, pos = self.samples[idx]
        negs = []
        user_pos_set = self.user_pos.get(u, set())

        while len(negs) < self.neg_k:
            n = random.randint(0, self.num_items - 1)
            if n not in user_pos_set:
                negs.append(n)

        return int(u), int(pos), torch.tensor(negs, dtype=torch.long)

def sampled_softmax_loss(u_ids, pos_i_ids, neg_i_ids, user_emb, item_emb):
    u = user_emb[u_ids]
    pos = item_emb[pos_i_ids]
    neg = item_emb[neg_i_ids]

    pos_logit = (u * pos).sum(dim=1, keepdim=True)
    neg_logits = torch.einsum("bd,bkd->bk", u, neg)
    logits = torch.cat([pos_logit, neg_logits], dim=1)
    targets = torch.zeros(u.shape[0], dtype=torch.long, device=u_ids.device)
    return F.cross_entropy(logits, targets)

def bpr_loss(u_ids, pos_i_ids, neg_i_ids, user_emb, item_emb):
    u = user_emb[u_ids]
    pos = item_emb[pos_i_ids]
    neg = item_emb[neg_i_ids]
    x = (u * pos).sum(dim=1) - (u * neg).sum(dim=1)
    return -F.logsigmoid(x).mean()

# =====================
# Training
# =====================

def train_lightgcn_with_aug(
    train_dict, aug_triplets, num_users_total, num_items_total,
    embedding_dim=64, num_layers=3, neg_k=10, epochs=10, batch_size=2048,
    lr=1e-3, dropout=0.0, device=None, lambda_aug=1.0
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LightGCN(
        num_users=num_users_total,
        num_items=num_items_total,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    adj_t = build_norm_adj(train_dict, num_users_total, num_items_total).to(device)

    main_ds = MainTrainDataset(train_dict, num_items=num_items_total, neg_k=neg_k)
    main_loader = DataLoader(
        main_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        worker_init_fn=seed_worker
    )

    if len(aug_triplets) > 0:
        aug_U = torch.tensor([t[0] for t in aug_triplets], dtype=torch.long, device=device)
        aug_P = torch.tensor([t[1] for t in aug_triplets], dtype=torch.long, device=device)
        aug_N = torch.tensor([t[2] for t in aug_triplets], dtype=torch.long, device=device)
    else:
        aug_U = aug_P = aug_N = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        for u, pos, negs in main_loader:
            u, pos, negs = u.to(device), pos.to(device), negs.to(device)
            user_emb, item_emb = model(adj_t)

            loss_main = sampled_softmax_loss(u, pos, negs, user_emb, item_emb)

            loss_bpr = torch.tensor(0.0, device=device)
            if aug_U is not None and aug_U.numel() > 0:
                idx = torch.randint(0, aug_U.shape[0], (u.shape[0],), device=device)
                u_a, p_a, n_a = aug_U[idx], aug_P[idx], aug_N[idx]
                loss_bpr = bpr_loss(u_a, p_a, n_a, user_emb, item_emb)

            loss = loss_main + lambda_aug * loss_bpr
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        print(f"[Epoch {epoch:02d}] Loss: {total_loss / max(1, steps):.4f}")

    return model, adj_t

# =====================
# Main
# =====================

if __name__ == "__main__":
    try_count = 1
    parts = 1
    alpha = 1
    random_seeds = [42, 2024, 7, 1234, 9999]

    for tr_cnt in range(try_count):
        run_dir = get_run_dir(alpha)
        os.makedirs(run_dir, exist_ok=True)

        ground_truth, date2idx = get_initial_data(run_dir, DATASET_NAME)

        predict_json_path = os.path.join(run_dir, f"predict_label_part{parts}_alpha{alpha}.json")
        if not os.path.exists(predict_json_path):
            with open(predict_json_path, "w") as f:
                json.dump({}, f)

        all_timestamps = sorted({ts for interactions in ground_truth.values() for _, ts in interactions})
        if len(all_timestamps) == 0:
            raise RuntimeError("ground_truth의 timestamp가 비어 있습니다.")
        base_part_size = max(1, len(all_timestamps) // parts)

        for step in range(parts):
            print(f"\n=== [{DATASET_NAME}] Feedback Loop Step {step+1}/{parts} ===")
            with open(predict_json_path, "r") as f:
                predict_label = json.load(f)

            label_dict, train_dict, warm_train, cold_items, n_users, n_items = get_data(run_dir)
            print(f"Cold items for augmentation: {len(cold_items)} / {n_items}")

            aug_triplets, count = augment_data(
                train_dict,
                cold_items,
                pairs_per_user=1,
                rng_seed=random_seeds[step],
                dataset_name=DATASET_NAME
            )

            with open(f"{run_dir}/aug_triplets_part{parts}_step{step}.pkl", "wb") as f:
                pickle.dump(aug_triplets, f)
            
            # LLM Stats 출력
            print(f"LLM Stats — A: {count[0]}, B: {count[1]}, Hallucination: {count[2]}")
            # LLM Stats 저장
            with open(f"{run_dir}/llm_stats_part{parts}_step{step}_alpha{alpha}.json", "w") as f:
                json.dump({
                    "count_a": count[0],
                    "count_b": count[1],
                    "hallucination": count[2]
                }, f, indent=2)
            
            # # load aug_triplets for verification
            # with open(f"{run_dir}/aug_triplets_part{parts}_step{step}.pkl", "rb") as f:
            #     aug_triplets = pickle.load(f)

            model, adj_t = train_lightgcn_with_aug(
                train_dict=train_dict,
                aug_triplets=aug_triplets,
                num_users_total=n_users,
                num_items_total=n_items,
                embedding_dim=64,
                num_layers=2,
                neg_k=10,
                epochs=30,
                batch_size=4096,
                lambda_aug=1.0
            )

            with torch.no_grad():
                user_emb, item_emb = model(adj_t)
                np.save(f"{run_dir}/user_emb_part{parts}_step{step}.npy", to_numpy_cpu(user_emb))
                np.save(f"{run_dir}/item_emb_part{parts}_step{step}.npy", to_numpy_cpu(item_emb))

            start_idx = step * base_part_size
            end_idx = (step + 1) * base_part_size if step < parts - 1 else len(all_timestamps)
            time_range = set(all_timestamps[start_idx:end_idx])
            if len(time_range) == 0:
                print("No timestamps in this step; skipping prediction update.")
                continue

            active_users: List[int] = [
                u for u, interactions in ground_truth.items()
                if any(ts in time_range for _, ts in interactions)
            ]
            active_users_set = set(active_users)

            device = user_emb.device
            temp_predict_label = defaultdict(list)

            for u, _ in label_dict.items():
                if u not in active_users_set:
                    continue

                Kmax = sum(1 for _, ts in ground_truth.get(u, []) if ts in time_range)
                if Kmax <= 0:
                    continue

                with torch.no_grad():
                    scores = (user_emb[u:u+1] @ item_emb.T).squeeze(0)

                    seen = train_dict.get(u, [])
                    if seen:
                        seen_idx = torch.tensor(seen, dtype=torch.long, device=device)
                        scores.index_fill_(0, seen_idx, -1e9)

                    num_unseen = scores.numel() - (len(seen) if seen else 0)
                    k = max(0, min(Kmax, num_unseen))
                    if k == 0:
                        continue

                    topk_idx = torch.topk(scores, k=k).indices.tolist()
                    seen_set = set(train_dict.get(u, []))
                    temp_recommended_items = [int(x) for x in topk_idx if x not in seen_set][:Kmax]

                    n_rec = int(round(Kmax * alpha))
                    n_rec = min(n_rec, Kmax)
                    recommended_items = temp_recommended_items[:n_rec]
                    feedback_existing_items = set(recommended_items)

                    gt_items_in_window = [
                        int(item)
                        for item, ts in ground_truth.get(u, [])
                        if ts in time_range
                    ]
                    gt_candidate_pool = [
                        item
                        for item in gt_items_in_window
                        if item not in seen_set and item not in feedback_existing_items
                    ]

                    remaining = Kmax - n_rec
                    rng = random.Random(random_seeds[step] + parts * 100000 + step * 1000 + int(u))
                    sampled_gt_items = rng.sample(
                        gt_candidate_pool,
                        k=min(remaining, len(gt_candidate_pool))
                    )

                    final_items = recommended_items + sampled_gt_items
                    if len(final_items) < Kmax:
                        final_existing_items = set(final_items)
                        for item in temp_recommended_items[n_rec:]:
                            if item in final_existing_items:
                                continue
                            final_items.append(item)
                            final_existing_items.add(item)
                            if len(final_items) >= Kmax:
                                break

                    final_items = final_items[:Kmax]
                    rng.shuffle(final_items)

                    new_items = [int(x) for x in final_items if x not in seen_set]
                    if new_items:
                        train_dict.setdefault(u, []).extend(new_items)

                setdefault_list(predict_label, u).extend(final_items)
                temp_predict_label[str(u)].extend(temp_recommended_items)

            with open(os.path.join(run_dir, "train.json"), "w") as f:
                json.dump({str(k): v for k, v in train_dict.items()}, f)
            with open(predict_json_path, "w") as f:
                json.dump(predict_label, f)

            temp_predict_path = os.path.join(
                run_dir,
                f"predict_label_part{parts}_step{step+1}.json"
            )
            with open(temp_predict_path, "w") as f:
                json.dump({str(k): v for k, v in temp_predict_label.items()}, f, indent=2)

            temp_predict_alpha_path = os.path.join(
                run_dir,
                f"predict_label_part{parts}_step{step+1}_alpha{alpha}.json"
            )
            with open(temp_predict_alpha_path, "w") as f:
                json.dump({str(k): v for k, v in temp_predict_label.items()}, f, indent=2)

            print(f"✅ Step {step} complete. Updated train.json")
            #print(f"✅ Saved temp recommendations without GT: {temp_predict_path}")
            print(f"✅ Saved temp recommendations without GT: {temp_predict_alpha_path}")
