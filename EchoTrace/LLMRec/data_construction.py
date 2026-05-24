import os
import json
import pandas as pd
import numpy as np
import scipy.sparse as sp
import pickle
from collections import defaultdict

def get_train(df):
        data = defaultdict(list)
        for user_id, group in df.groupby("user_id"):
            items = list(zip(group["item_id"], group["rating"]))
            for item in items:
                data[user_id].append(item[0])
        return data

def get_gt(test_path = None):
    # 1. 날짜 문자열 추출
    df = pd.read_csv(test_path, sep=r"\s+", header=None, names=["user_id", "item_id", "timestamp"])
    
    # 만약 유닉스 타임스탬프라면 변환 
    if df["timestamp"].dtype != object:
        df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime('%Y-%m-%d')
    else:
        df["date"] = df["timestamp"]  # 이미 문자열 날짜라고 가정
    # 2. 고유 날짜를 정렬 후, 0부터 정수 매핑
    unique_dates = sorted(df["date"].unique())
    date2idx = {date: idx for idx, date in enumerate(unique_dates)}

    # 3. 사용자별 (item_id, date_idx) 리스트 생성
    data = defaultdict(list)
    for user_id, group in df.groupby("user_id"):
        items = list(zip(group["item_id"], group["date"]))
        for item in items:
            item_id, date = item
            date_idx = date2idx[date]  # 정수 매핑
            data[user_id].append((item_id, date_idx))
    
    return data, date2idx  # ✅ date2idx도 반환

def get_data(file_path="./data/books", folder="books_llmrec_format", json_indent=None):
    """
    Books: u_data_2017_kcore10_user_item_split_filtered.txt를 읽어서
    MovieLens와 동일한 LLMRec 입력 포맷(train/test.json)으로 변환
    """
    
    test_path = os.path.join(file_path, "label.txt")
    train_path = os.path.join(file_path, "train.txt")
    train_data = {}
    with open(train_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u_str, i_str = line.split()[:2]
            
            u = str(u_str)         # 키는 문자열로
            i = int(i_str)         # 아이템은 정수로 (혹은 문자열로 둘 수도 있음)

            if u not in train_data:
                train_data[u] = []
            train_data[u].append(i)

    test_data = {}
    with open(test_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u_str, i_str = line.split()[:2]
            u = str(u_str)
            i = int(i_str)
            # ground truth가 여러 개면 리스트에 append, 한 개만 있으면 덮어쓰거나 첫 개만 쓰면 됨
            if u not in test_data:
                test_data[u] = []
            test_data[u].append(i)
            
    # ===== Build train/test =====
    test_data_timestamp, date2idx = get_gt(test_path)

    # ===== Save JSON =====
    out_dir = f"{file_path}/{folder}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(os.path.join(out_dir, "train.json"), "w") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=json_indent)
    with open(os.path.join(out_dir, "test.json"), "w") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=json_indent)

    
    print("✅ Saved train.json, test.json")
    if json_indent is not None:
        print(f"json_indent: {json_indent}")
    print(f"train_path: {os.path.join(out_dir, 'train.json')}")
    print(f"test_path: {os.path.join(out_dir, 'test.json')}")

    return test_data_timestamp, date2idx

def get_train_matrix(file_path = "./data/books", folder="books_llmrec_format"):
    # ====== Load train/val/test JSONs ======
    data_path = f"{file_path}/{folder}/"
    with open(os.path.join(data_path, "train.json"), "r") as f:
        train_dict = json.load(f)
    with open(os.path.join(data_path, "test.json"), "r") as f:
        test_dict = json.load(f)
    # ====== Get n_users, n_items ======
    all_user_ids = set(map(int, train_dict.keys())) | set(map(int, test_dict.keys()))
    all_item_ids = set()
    for d in [train_dict, test_dict]:
        for items in d.values():
            all_item_ids.update(items)

    n_users = len(all_user_ids)
    n_items = len(all_item_ids)
    print(f"n_users={n_users}, n_items={n_items}")

    # ====== Build sparse matrix ======
    def build_sparse_matrix(data_dict, n_users, n_items):
        mat = sp.dok_matrix((n_users, n_items), dtype=np.float32)
        for user, items in data_dict.items():
            for item in items:
                if int(item) < n_items:
                    mat[int(user), int(item)] = 1.0
        return mat.tocsr()

    train_mat = build_sparse_matrix(train_dict, n_users, n_items)
    test_mat = build_sparse_matrix(test_dict, n_users, n_items)
    
    # ====== Save matrices as pickle ======
    with open(os.path.join(data_path, "train_mat"), "wb") as f:
        pickle.dump(train_mat, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(data_path, "test_mat"), "wb") as f:
        pickle.dump(test_mat, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("✅ Saved train_mat, test_mat as .pkl files")
    return n_users, n_items
    
    
def main():
    # ====== Config ======
    data_path = "./data/yelp"
    folder = "yelp_llmrec_format"
    ground_truth, date2idx = get_data(data_path, folder, json_indent=2)
    train_mat, test_mat = get_train_matrix(data_path, folder)
    
    print("finished")


if __name__ == "__main__":
    main()
    
