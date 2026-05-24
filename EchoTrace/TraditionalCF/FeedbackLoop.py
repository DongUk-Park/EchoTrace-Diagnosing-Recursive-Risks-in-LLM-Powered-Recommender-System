#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
feedback_loop.py — Yelp (Pre-split files) 기반 전통 CF 피드백 루프
- u.data 분할 로직 제거 -> train.txt, label.txt 직접 로드
- train.txt -> train.json (초기 학습용)
- label.txt -> label.json (Ground Truth) 및 date_idx 윈도우 생성
"""

from __future__ import annotations
import os, sys, json, random, argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd
import torch

# -------- 고정 경로 / 상수 --------
dataset = "yelp"  # "ml-1m", "books", "yelp"
DATA_ROOT = f"./data/{dataset}"

TRAIN_RAW_FILE = "train.txt"                      # DATA_ROOT/train.txt (u i timestamp)
LABEL_RAW_FILE = "label.txt"                      # DATA_ROOT/label.txt (u i timestamp)

TRAIN_REL  = "traditionalCF/train.json"           # DATA_ROOT/traditionalCF/train.json
LABEL_REL  = "traditionalCF/label.json"           # DATA_ROOT/traditionalCF/label.json
PRED_DIR   = "traditionalCF"                      # 예측 기록 저장 디렉토리
RESULTS_DIR = os.path.join(DATA_ROOT, PRED_DIR)

# 데이터셋 크기 (코드 실행 시 max id로 갱신됨)
NUM_ITEMS_TOTAL = 0 
NUM_USERS_TOTAL = 0
PARTS = 1
ALPHA = 1

# LightGCN 모듈 import 경로 (환경에 맞게 수정)
sys.path.append("./TraditionalCF")
from LightGCN import train_lightgcn_softmax

# ------------------
# Utils
# ------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _ensure_int_dict(d: Dict) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for k, v in d.items():
        try: ki = int(k)
        except: continue
        if isinstance(v, list):
            out[ki] = [int(x) for x in v]
        else:
            try: out[ki] = [int(v)]
            except: out[ki] = []
    return out

def _save_json(path: str, d: Dict[int, List[int]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in d.items()}, f)

def _load_json(path: str) -> Dict[int, List[int]]:
    with open(path, "r") as f:
        raw = json.load(f)
    return _ensure_int_dict(raw)

# ------------------
# 데이터 로드 및 초기화
# ------------------
def get_gt(df: pd.DataFrame):
    """label_df에서 (user -> [(item_id, date_idx), ...])와 date2idx 제공."""
    df = df.copy()
    if df["timestamp"].dtype != object:
        df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime('%Y-%m-%d')
    else:
        df["date"] = df["timestamp"]  # 이미 문자열 날짜라고 가정
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

def get_initial_data_from_files():
    """
    DATA_ROOT 내의 train.txt, label.txt를 직접 읽어서 초기화
    - train.txt -> train.json
    - label.txt -> label.json & Loop용 Date info
    """
    train_raw_path = os.path.join(DATA_ROOT, TRAIN_RAW_FILE)
    label_raw_path = os.path.join(DATA_ROOT, LABEL_RAW_FILE)
    
    if not os.path.exists(train_raw_path) or not os.path.exists(label_raw_path):
        raise FileNotFoundError(f"Check files: {train_raw_path}, {label_raw_path}")

    # Load DataFrames (sep=None to auto-detect space/tab)
    print(f"Loading raw files from {DATA_ROOT}...")
    train_df = pd.read_csv(train_raw_path, sep=r"\s+", names=["user_id", "item_id", "timestamp"])
    label_df = pd.read_csv(label_raw_path, sep=r"\s+", names=["user_id", "item_id", "timestamp"])

    # 전역 User/Item 수 갱신 (max ID + 1)
    global NUM_USERS_TOTAL, NUM_ITEMS_TOTAL
    max_u = max(train_df["user_id"].max(), label_df["user_id"].max())
    max_i = max(train_df["item_id"].max(), label_df["item_id"].max())
    NUM_USERS_TOTAL = int(max_u) + 1
    NUM_ITEMS_TOTAL = int(max_i) + 1
    print(f"Detected Stats: Users={NUM_USERS_TOTAL}, Items={NUM_ITEMS_TOTAL}")

    # JSON 저장 경로
    train_json_path = os.path.join(DATA_ROOT, TRAIN_REL)
    label_json_path = os.path.join(DATA_ROOT, LABEL_REL)

    # train.json 생성
    train_dict: Dict[int, List[int]] = {int(uid): [int(x) for x in g["item_id"].tolist()]
                                        for uid, g in train_df.groupby("user_id")}
    _save_json(train_json_path, train_dict)

    # label.json 생성
    label_dict: Dict[int, List[int]] = {int(uid): [int(x) for x in g["item_id"].tolist()]
                                        for uid, g in label_df.groupby("user_id")}
    _save_json(label_json_path, label_dict)

    print(f"✅ Saved train.json({len(train_df)}) & label.json({len(label_df)}) at {os.path.dirname(train_json_path)}")

    # Feedback Loop용 Date Parsing
    test_data_timestamp, date2idx = get_gt(label_df)
    return test_data_timestamp, date2idx

# ------------------
# 추천 업데이트 (윈도우 = date_idx 구간)
# ------------------
@torch.no_grad()
def recommend_and_update_by_datewindow(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    train_dict: Dict[int, List[int]],
    gt_by_dateidx: Dict[int, List[Tuple[int, int]]],
    date_idx_range: Set[int],
    alpha: float,
) -> Dict[int, List[int]]:
    
    device = user_emb.device
    pred_dict: Dict[int, List[int]] = {}
    seen_cache: Dict[int, set] = {u: set(items) for u, items in train_dict.items()}

    # 이번 윈도우에 등장한 Active User
    active_users = [u for u, pairs in gt_by_dateidx.items()
                    if any(di in date_idx_range for _, di in pairs)]

    for u in active_users:
        interactions_in_window = [
            (int(it), di) for (it, di) in gt_by_dateidx.get(u, [])
            if di in date_idx_range
        ]
        interactions_in_window.sort(key=lambda x: x[1])

        # need: 이번 윈도우 실제 인터랙션 수
        need = len(interactions_in_window)
        if need <= 0:
            pred_dict[u] = []
            continue

        n_rec = min(int(round(need * alpha)), need)
        remaining = need - n_rec
        seen_set = seen_cache.get(u, set())

        # Score 계산
        scores = (user_emb[u:u+1] @ item_emb.T).squeeze(0)  # [num_items_total]
        
        # Seen Masking
        seen = list(seen_set)
        if seen:
            seen_idx = torch.tensor(seen, dtype=torch.long, device=device)
            scores.index_fill_(0, seen_idx, -1e9)

        num_unseen = int((scores > -1e9/2).sum().item())
        k = max(0, min(n_rec, num_unseen))
        recommended_ids = []
        if k > 0:
            recommended_ids = [
                int(x) for x in torch.topk(scores, k=k).indices.tolist()
                if int(x) not in seen_set
            ]

        selected_set = set(recommended_ids)
        gt_candidate_pool = [
            int(item_id) for item_id, _ in interactions_in_window
            if int(item_id) not in seen_set and int(item_id) not in selected_set
        ]
        sampled_gt_ids = random.sample(
            gt_candidate_pool,
            k=min(remaining, len(gt_candidate_pool))
        )

        final_ids = recommended_ids + sampled_gt_ids

        # Train 업데이트 (Feedback)
        if final_ids:
            train_dict.setdefault(u, []).extend(final_ids)
            seen_set.update(final_ids)
            seen_cache[u] = seen_set

        pred_dict[u] = final_ids

    return pred_dict


# ------------------
# Main
# ------------------
def main():
    parser = argparse.ArgumentParser(description="LightGCN Feedback Loop for Yelp (Pre-split)")
    parser.add_argument("--parts", type=int, default=PARTS, help="피드백 루프 스텝 수")
    parser.add_argument("--epochs", type=int, default=30, help="LightGCN 학습 epoch 수")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=ALPHA,
                        help="각 피드백 스텝에서 추천 item으로 채울 비율. 나머지는 GT에서 샘플링")
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1")

    set_seed(args.seed)
    
    # 1) 초기 데이터 로드 (파일 기반) & 전역 User/Item 수 설정
    print("Initializing Data from train.txt / label.txt ...")
    ground_truth, date2idx = get_initial_data_from_files()

    # 2) train.json 다시 로드 (초기화 확인)
    train_path = os.path.join(DATA_ROOT, TRAIN_REL)
    train_dict = _load_json(train_path)

    # 3) Date Index Timeline
    all_timestamps = sorted({ts for interactions in ground_truth.values() for _, ts in interactions})
    if len(all_timestamps) == 0:
        raise RuntimeError("ground_truth의 timestamp가 비어 있습니다.")
    base_part_size = max(1, len(all_timestamps) // args.parts)
    predict_label = defaultdict(list)

    # 4) Feedback Loop
    for step in range(args.parts):
        print(f"\n=== Feedback Step {step+1}/{args.parts} ===")
        
        # 최신 train 로드
        train_dict = _load_json(train_path)
        # 빈 유저 보정 (GCN 행렬 크기 맞춤): 근데 test에만 존재하는 유저는 없기때문에 걸리는 케이스는 없음
        for u in range(NUM_USERS_TOTAL):
            if u not in train_dict:
                train_dict[u] = []
                
        # 학습
        model, adj_t = train_lightgcn_softmax(
            train_dict=train_dict,
            embedding_dim=args.embedding_dim,
            num_layers=args.num_layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            dropout=args.dropout,
            neg_k=10,
            num_items_total=NUM_ITEMS_TOTAL, # LightGCN에 전달
            num_users_total=NUM_USERS_TOTAL
        )
        
        # 임베딩 추출
        with torch.no_grad():
            user_emb, item_emb = model(adj_t)
        
        # 저장
        os.makedirs(RESULTS_DIR, exist_ok=True)
        np.save(os.path.join(RESULTS_DIR, f"user_emb_part{args.parts}_step{step}.npy"), user_emb.detach().cpu().numpy())
        np.save(os.path.join(RESULTS_DIR, f"item_emb_part{args.parts}_step{step}.npy"), item_emb.detach().cpu().numpy())

        # Date Window 계산
        start_idx = step * base_part_size
        end_idx = (step + 1) * base_part_size if step < args.parts - 1 else len(all_timestamps)
        date_idx_range = set(all_timestamps[start_idx:end_idx])
        
        if not date_idx_range:
            print("[INFO] No dates in this window, skipping update.")
            continue

        # 추천 및 업데이트
        pred = recommend_and_update_by_datewindow(
            user_emb=user_emb,
            item_emb=item_emb,
            train_dict=train_dict,
            gt_by_dateidx=ground_truth,
            date_idx_range=date_idx_range,
            alpha=args.alpha,
        )

        for u, items in pred.items():
            if items:
                predict_label[u].extend(items)
        
        # 갱신된 Train 저장
        _save_json(train_path, train_dict)
        
        # Step별 Log 저장
        alpha_tag = f"{args.alpha:g}"
        pred_path = os.path.join(RESULTS_DIR, f"predict_label_part{args.parts}_step{step}_alpha{alpha_tag}.json")
        with open(pred_path, "w") as f:
            json.dump({str(k): v for k, v in pred.items()}, f, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"[STEP {step}] Recommendations saved & train.json updated.")
    
    # 전체 기록 저장
    alpha_tag = f"{args.alpha:g}"
    final_path = os.path.join(RESULTS_DIR, f"predict_label_part{args.parts}_alpha{alpha_tag}.json")
    with open(final_path, "w") as f:
        json.dump({str(k): v for k, v in predict_label.items()}, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"Final Loop Completed. Result -> {final_path}")

if __name__ == "__main__":
    main()
