#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
import json
import random
import argparse
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Any, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import HashingVectorizer

import LLMRec
import data_construction
from LLM_augmentation_construct_prompt import main_generate
import LightGCN


# ------------------
# Utils
# ------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _ensure_list_int(v: Any) -> List[int]:
    if v is None:
        return []
    if isinstance(v, list):
        out = []
        for x in v:
            try:
                out.append(int(x))
            except Exception:
                continue
        return out
    try:
        return [int(v)]
    except Exception:
        return []


def _load_json(path: str) -> Dict[str, List[int]]:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        raw = json.load(f)
    out: Dict[str, List[int]] = {}
    for k, v in raw.items():
        out[str(k)] = _ensure_list_int(v)
    return out


def _save_json(path: str, d: Dict[str, List[int]], *, indent: Optional[int] = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f, ensure_ascii=False, indent=indent, sort_keys=True)


def _ensure_yelp_llmrec_assets(file_path: str, folder: str) -> None:
    if os.path.basename(file_path) != "yelp":
        return

    base_dir = os.path.join(file_path, folder)
    os.makedirs(base_dir, exist_ok=True)

    item_attr_path = os.path.join(base_dir, "item_attribute.csv")
    text_feat_path = os.path.join(base_dir, "text_feat.npy")
    image_feat_path = os.path.join(base_dir, "image_feat.npy")

    if os.path.exists(item_attr_path) and os.path.exists(text_feat_path) and os.path.exists(image_feat_path):
        return

    meta_path = os.path.join(file_path, "item_meta_2018_kcore5_user_item_split_filtered.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Yelp item meta file not found: {meta_path}")

    meta = pd.read_json(meta_path, lines=True)
    df = meta[["item_id", "name", "address", "city", "state", "categories", "stars", "review_count"]].copy()
    df = df.rename(columns={"item_id": "id"})
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    df = df.sort_values("id")
    df = df.fillna("unknown")

    if not os.path.exists(item_attr_path):
        df.to_csv(item_attr_path, index=False, header=False)
        print(f"✅ Saved Yelp item_attribute.csv: {item_attr_path}")

    n_items = len(df)
    if not os.path.exists(text_feat_path):
        text = (
            df["name"].astype(str) + " " +
            df["address"].astype(str) + " " +
            df["city"].astype(str) + " " +
            df["state"].astype(str) + " " +
            df["categories"].astype(str) + " stars " +
            df["stars"].astype(str) + " review_count " +
            df["review_count"].astype(str)
        )
        vectorizer = HashingVectorizer(n_features=1536, alternate_sign=False, norm="l2")
        text_feat = vectorizer.transform(text).astype(np.float32).toarray()
        np.save(text_feat_path, text_feat)
        print(f"✅ Saved Yelp text_feat.npy: {text_feat_path}, shape={text_feat.shape}")

    if not os.path.exists(image_feat_path):
        np.save(image_feat_path, np.zeros((n_items, 1536), dtype=np.float32))
        print(f"✅ Saved Yelp zero image_feat.npy: {image_feat_path}, shape=({n_items}, 1536)")


def _count_interactions_gt(ground_truth: Dict[int, List[Tuple[int, int]]]) -> int:
    return int(sum(len(v) for v in ground_truth.values()))


def _count_interactions_pred(pred: Dict[str, List[int]]) -> int:
    return int(sum(len(v) for v in pred.values()))


def _get_all_timestamps(ground_truth: Dict[int, List[Tuple[int, int]]]) -> List[int]:
    return sorted({ts for interactions in ground_truth.values() for _, ts in interactions})


def _active_users_in_time_range(
    ground_truth: Dict[int, List[Tuple[int, int]]],
    time_range: Set[int],
) -> List[int]:
    return [
        user for user, interactions in ground_truth.items()
        if any(ts in time_range for _, ts in interactions)
    ]


def _truncate_candidates_by_gt_count(
    best_candidates_full: Dict[int, List[int]],
    ground_truth: Dict[int, List[Tuple[int, int]]],
    active_users: List[int],
    time_range: Set[int],
    alpha: float = 0.5,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    각 active user에 대해:
    - 현재 window의 GT 개수 = k
    - 최종 길이도 k로 맞춤
    - temp 추천 결과는 GT를 섞지 않고 추천 후보에서 k개까지 저장
    - 그중 alpha 비율은 추천 결과(best_candidates)에서 채움
    - 나머지 (1-alpha)는 현재 window의 GT item들에서 랜덤 샘플링하여 채움
    - GT 보충 이후 최종 결과를 shuffle

    주의:
    - 추천 결과와 GT 보충 결과 간 중복 제거
    - 추천 후보가 부족하면 GT로 추가 보충
    """
    out: Dict[int, List[int]] = {}
    temp_out: Dict[int, List[int]] = {}

    for u in active_users:
        # 현재 window 안의 GT item 목록
        gt_items_in_window = [int(it) for it, ts in ground_truth[u] if ts in time_range]
        k = len(gt_items_in_window)

        if k <= 0:
            out[u] = []
            temp_out[u] = []
            continue

        # 추천으로 채울 목표 개수
        n_rec = int(round(k * alpha))
        n_rec = min(n_rec, k)

        # 추천 후보 가져오기
        rec_candidates = best_candidates_full.get(u, [])

        temp_recommended = []
        temp_recommended_set = set()
        for item in rec_candidates:
            item = int(item)
            if item in temp_recommended_set:
                continue
            temp_recommended.append(item)
            temp_recommended_set.add(item)
            if len(temp_recommended) >= k:
                break

        chosen = []
        chosen_set = set()

        # 1) 추천 결과에서 먼저 채우기
        for item in rec_candidates:
            item = int(item)
            if item in chosen_set:
                continue
            chosen.append(item)
            chosen_set.add(item)
            if len(chosen) >= n_rec:
                break

        # 2) 남은 자리는 GT에서 랜덤 샘플링
        remaining = k - n_rec

        gt_pool = []
        for item in gt_items_in_window:
            if item not in chosen_set:
                gt_pool.append(item)

        if len(gt_pool) > 0 and remaining > 0:
            sampled_gt = random.sample(gt_pool, k=min(remaining, len(gt_pool)))
            for item in sampled_gt:
                if item not in chosen_set:
                    chosen.append(item)
                    chosen_set.add(item)

        # 3) 혹시 추천 부족 + GT 중복 등으로 길이가 k보다 짧으면,
        #    추천 후보에서 남은 것 다시 채움
        if len(chosen) < k:
            for item in rec_candidates:
                item = int(item)
                if item in chosen_set:
                    continue
                chosen.append(item)
                chosen_set.add(item)
                if len(chosen) >= k:
                    break

        # 최종적으로 길이 k까지만 사용
        final_items = chosen[:k]
        random.shuffle(final_items)

        out[u] = final_items
        temp_out[u] = temp_recommended

    return out, temp_out


def _update_train_and_predict_json(
    json_path: str,
    predict_json_path: str,
    best_candidates_t: Dict[int, List[int]],
) -> None:
    """
    기존 update_train_json_with_candidates와 동일한 효과:
    - train.json: user별로 candidates append
    - predict_label.json: user별로 candidates append
    - (중복 제거/seen masking 없음: 기존과 동일)
    """
    train_dict = _load_json(json_path)
    predict_label = _load_json(predict_json_path)

    for user_id, items in best_candidates_t.items():
        user_id_str = str(user_id)

        if user_id_str not in train_dict:
            train_dict[user_id_str] = []
        train_dict[user_id_str].extend(items)

        if user_id_str not in predict_label:
            predict_label[user_id_str] = []
        predict_label[user_id_str].extend(items)

    _save_json(json_path, train_dict, indent=None)
    _save_json(predict_json_path, predict_label, indent=None)


# ------------------
# Main Loop
# ------------------
def feedback_loop():
    dataset_name = "ml-1m"  # netflix, ml-1m, books, yelp
    file_path = f"./data/{dataset_name}"
    folder = f"{dataset_name}_llmrec_format"
    parts = 5
    alpha = 1
    print(f"Running LLMRec Feedback Loop with dataset={dataset_name}, parts={parts}, alpha={alpha}")
    parser = argparse.ArgumentParser(description="LLMRec Feedback Loop (refactored like traditionalCF script)")
    parser.add_argument("--file_path", type=str, default=file_path)
    parser.add_argument("--folder", type=str, default=folder)
    parser.add_argument("--dataset_name", type=str, default=dataset_name)  # netflix, ml-1m, books, yelp
    parser.add_argument("--parts", type=int, default=parts)                            
    parser.add_argument("--alpha", type=float, default=alpha)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    print("ENV CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))

    # ---------- Paths ----------   
    _ensure_yelp_llmrec_assets(args.file_path, args.folder)
    base_dir = os.path.join(args.file_path, args.folder)
    train_json_path = os.path.join(base_dir, "train.json")
    predict_json_path = os.path.join(base_dir, f"predict_label_part{args.parts}.json")
    print(f"train_json_path: {train_json_path}")
    print(f"predict_json_path: {predict_json_path}")


    # ---------- 1) 초기 데이터 로드 ----------
    ground_truth, date2idx = data_construction.get_data(args.file_path, args.folder)
    
    # 1-1) train.json을 초기화하지 않고, 기존 데이터 그대로 사용 (기존 동작과 동일)
    # test_path = os.path.join(file_path, "label.txt")
    # ground_truth, date2idx = data_construction.get_gt(test_path)

    expected = _count_interactions_gt(ground_truth)
    print("expected ground_truth interactions:", expected)

    # ---------- 2) predict_label.json 초기화(기존 동작 그대로: 매 실행 시 새로 생성) ----------
    print("📁 Creating new predict_label.json")
    _save_json(predict_json_path, {}, indent=None)

    # ---------- 3) time window 구성 ----------
    all_timestamps = _get_all_timestamps(ground_truth)
    if len(all_timestamps) == 0:
        raise RuntimeError("ground_truth timestamps is empty.")

    part_size = len(all_timestamps) // args.parts if args.parts > 0 else len(all_timestamps)
 
    # ---------- 4) Feedback Loop ----------
    for t in range(args.parts):
        print(f"\n🚀 Feedback Loop - Time Step t = {t}")

        # Step: train/test.json -> train/test.mat 생성 (기존 그대로)
        n_users, n_items = data_construction.get_train_matrix(args.file_path, args.folder)
        print(f"train_mat: n_users={n_users}, n_items={n_items}")
        
        if t >= 1:            
            LightGCN.main(args.file_path, args.folder) # candidate 생성
            profile_step_num = None # t / None
            profile_parts = None # args.parts / None
            
            main_generate.main(
                args.dataset_name,
                file_path=base_dir,
                profile_step_num=profile_step_num, #profile_step_num,
                parts=profile_parts, #profile_parts,
                # 둘 다 None이면 프로필 생성
            )

            if profile_step_num is None and profile_parts is None:
                profiling_path = os.path.join(base_dir, "augmented_user_profiling_dict")
                save_path = os.path.join(
                    base_dir,
                    f"augmented_user_profiling_dict_part{args.parts}_step{t}_alpha{args.alpha}",
                )
                augmented_user_profiling_dict = pickle.load(open(profiling_path, "rb"))
                with open(save_path, "wb") as f:
                    pickle.dump(augmented_user_profiling_dict, f)
                print(f"Saved generated user profiling dict: {save_path}")

        # LLMRec 실행 (기존 그대로)
        best_candidates_tensor, ua_embeddings, ia_embeddings = LLMRec.main(args.file_path, args.folder)

        ua_np = ua_embeddings.detach().cpu().numpy()
        ia_np = ia_embeddings.detach().cpu().numpy()

        np.save(os.path.join(base_dir, f"user_emb_part{args.parts}_step{t}_alpha{args.alpha}.npy"), ua_np)
        np.save(os.path.join(base_dir, f"item_emb_part{args.parts}_step{t}_alpha{args.alpha}.npy"), ia_np)

        # tensor -> dict (기존 그대로: user_id range(shape[0]))
        best_candidates_full: Dict[int, List[int]] = {
            user_id: best_candidates_tensor[user_id].tolist()
            for user_id in range(best_candidates_tensor.shape[0])
        }

        # time window 계산 (기존 그대로)
        start_idx = t * part_size
        end_idx = (t + 1) * part_size if t < args.parts - 1 else len(all_timestamps)
        time_range = set(all_timestamps[start_idx:end_idx])

        # active users
        active_users = _active_users_in_time_range(ground_truth, time_range)

        # debug: missing users
        active_set = set(active_users)
        cand_set = set(best_candidates_full.keys())
        missing = sorted(active_set - cand_set)
        print(f"[DEBUG] active_users={len(active_set)}, candidates_users={len(cand_set)}, missing_in_candidates={len(missing)}")
        if len(missing) > 0:
            print("[DEBUG] example missing users:", missing[:20])

        # GT interaction 수만큼 후보 구성:
        # - best_candidates_t: GT 보충 후 shuffle된 feedback용 후보
        # - temp_candidates_t: GT 없이 추천만 GT 길이만큼 저장할 후보
        best_candidates_t, temp_candidates_t = _truncate_candidates_by_gt_count(
            best_candidates_full=best_candidates_full,
            ground_truth=ground_truth,
            active_users=active_users,
            time_range=time_range,
            alpha=args.alpha,
        )
        # train.json + predict_label.json 업데이트 (기존 update_train_json_with_candidates와 동일 효과)
        _update_train_and_predict_json(
            json_path=train_json_path,
            predict_json_path=predict_json_path,
            best_candidates_t=best_candidates_t,
        )

        temp_predict_path = os.path.join(
            base_dir,
            f"predict_label_part{args.parts}_step{t+1}_alpha{args.alpha}.json",
        )
        _save_json(temp_predict_path, {str(k): v for k, v in temp_candidates_t.items()}, indent=2)
        print(f"✅ Saved temp recommendations without GT: {temp_predict_path}")

        # (기존 코드에서 찍던 new_time 로그는 실제로 저장/사용하지 않음)
        # 기존 함수는 max_time+part_size를 출력만 했으므로, 동일하게 로그만 출력해줌.
        max_time = max(ts for interactions in ground_truth.values() for _, ts in interactions)
        new_time = max_time + part_size
        print(f"✅ Updated train.json and predict_label.json for time {new_time}")

    # ---------- 5) 종료 후 predict_label 통계 출력 ----------
    pred = _load_json(predict_json_path)
    actual = _count_interactions_pred(pred)
    print("actual predict_label interactions:", actual)
    print("diff:", expected - actual)


if __name__ == "__main__":
    feedback_loop()
