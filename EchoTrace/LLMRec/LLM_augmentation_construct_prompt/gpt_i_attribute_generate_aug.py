import threading
import time
import pandas as pd
import pickle
import os
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

API_KEY = os.environ.get("OPENAI_API_KEY")
REQUEST_MAX_RETRIES = 3
REQUEST_TIMEOUT = 30
STEP1_MAX_PASSES = 3
PLACEHOLDER_RESPONSES = {
    ("director", "country", "language"),
    ("author", "country", "language"),
    ("publisher", "country", "language"),
    ("ambience", "visit_purpose", "companions_type"),
    ("ambience_type", "visit_purpose", "companions_type"),
}
PLACEHOLDER_FIELD_VALUES = {
    "director",
    "country",
    "language",
    "author",
    "publisher",
    "ambience",
    "ambience_type",
    "visit_purpose",
    "types of companions",
    "companions_type",
}
UNAVAILABLE_FIELD_VALUES = {
    "-",
    "n/a",
    "na",
    "none",
    "null",
    "not available",
    "unavailable",
    "unknown",
}

# --- Helper Functions ---

def load_item_attribute(file_path, dataset):
    attr_path = os.path.join(file_path, 'item_attribute.csv')
    cols_map = {
        "netflix": ['id', 'year', 'title'],
        "movielens": ['id', 'year', 'title', 'genre'],
        "books": ['id', 'brand', 'title', 'category'],
        "mind": ['id', 'title', 'category', 'subcategory'],
        "yelp": ['id', 'name', 'address', 'city', 'state', 'categories', 'stars', 'review_count']
    }
    return pd.read_csv(attr_path, names=cols_map[dataset.lower()])

def print_item_coverage(label, expected_count, actual_keys):
    actual_keys = {int(k) for k in actual_keys}
    expected_keys = set(range(expected_count))
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    print(
        f"{label}: expected_items={expected_count}, "
        f"covered_items={len(actual_keys & expected_keys)}, "
        f"missing={len(missing)}, extra={len(extra)}"
    )
    if missing:
        print(f"{label}: missing sample={missing[:20]}")
    if extra:
        print(f"{label}: extra sample={extra[:20]}")

def is_invalid_llm_values(values):
    lowered_values = tuple(v.strip().lower() for v in values)
    if any(not v for v in lowered_values):
        return True, "response contained empty :: separated fields"
    if lowered_values in PLACEHOLDER_RESPONSES:
        return True, "response copied the placeholder output format"
    if any(v in PLACEHOLDER_FIELD_VALUES for v in lowered_values):
        return True, "response contained placeholder field names"
    if all(v in UNAVAILABLE_FIELD_VALUES for v in lowered_values):
        return True, "response contained only unavailable values"
    return False, None

def construct_prompting(item_attribute, indices, dataset):
    index = indices[0]
    ds_lower = dataset.lower()
    
    # 데이터셋별 단수/복수 및 속성 정의 (프롬프트 문법 오류 방지)
    config = {
        "netflix": {
            "fields": ["year", "title"],
            "type_plural": "movies", "type_singular": "movie",
            "info": "director, country, language"
        },
        "movielens": {
            "fields": ["year", "title", "genre"],
            "type_plural": "movies", "type_singular": "movie",
            "info": "director, country, language"
        },
        "books": {
            "fields": ["brand", "title", "category"],
            "type_plural": "books", "type_singular": "book",
            "info": "author, country, language"
        },
        "yelp": {
            "fields": ["name", "address", "city", "state", "categories", "stars", "review_count"],
            "type_plural": "Yelp businesses", "type_singular": "Yelp business",
            "info": "ambience_type, visit_purpose, companions_type"
        }
    }

    cfg = config.get(ds_lower)
    if not cfg:
        raise ValueError(f"Unknown dataset: {dataset}")

    # DataFrame index 안전한 접근 (.iloc 사용)
    attr_values = [str(item_attribute[f].iloc[index]) for f in cfg["fields"]]
    item_info = ", ".join(f"{f}: {v}" for f, v in zip(cfg["fields"], attr_values))
    
    pre_string = (
        f"Provide the inquired information "
        f"of the given {cfg['type_plural']}."
        f"includes {', '.join(cfg['fields'])}:\n"
    )
    item_list_string = f"[{index}] {item_info}\n"
    
    target_format = cfg["info"].replace(", ", "::")
    output_format = (
        f"The inquired information is: {cfg['info']}.\n"
        f"Please output them in the following format:\n"
        f"{target_format}\n"
        f"Please output only the content in the format above, i.e., {target_format}.\n"
        f"Do not include reasoning, item index, or any extra text.\n\n"
    )
    
    return pre_string + item_list_string + output_format


def LLM_request_worker(index, toy_item_attribute, model_type, dataset):
    indices = [index]
    prompt = construct_prompting(toy_item_attribute, indices, dataset)
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    params = {
        "model": model_type,
        "messages": [
            {"role": "system", "content": "You are now a search engine."},
            {"role": "user", "content": prompt}
        ],
        "max_completion_tokens": 2048,
        #"temperature": 0.6,
        "stream": False
    }

    last_error = None
    for attempt in range(REQUEST_MAX_RETRIES):
        try:
            response = requests.post(url=url, headers=headers, json=params, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                usage = response.json().get("usage", {})
                print("usage:", usage)

                content = response.json()['choices'][0]['message']['content'].strip()
                print(content)
                rows = content.split("\n")

                # LLM이 마크다운(` ``` `)을 포함하거나 빈 줄을 출력할 경우를 대비해 robust하게 탐색
                for row in rows:
                    if "::" in row:
                        elements = [e.strip() for e in row.split("::")]
                        if len(elements) >= 3:
                            values = elements[:3]
                            invalid, reason = is_invalid_llm_values(values)
                            if invalid:
                                last_error = reason
                                continue
                            return index, {0: values[0], 1: values[1], 2: values[2]}
                last_error = "response did not contain three :: separated fields"
            else:
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"

            if response.status_code != 200 and response.status_code not in [408, 409, 429, 500, 502, 503, 504]:
                break
        except Exception as e:
            last_error = repr(e)

        time.sleep(min(60, 2 ** attempt))

    print(f"❌ Failed item {index} after {REQUEST_MAX_RETRIES} retries: {last_error}")
    print(f"⚠️ Filling item {index} with unknown attributes.")
    return index, {0: "unknown", 1: "unknown", 2: "unknown"}

def LLM_embedding_worker(index, row_data, keys, model_type):
    result_embeddings = {}
    MAX_RETRIES = 3
    BASE_TIMEOUT = 30
    
    for key in keys:
        text_input = str(row_data[key])
        if not text_input or text_input.lower() == 'nan': 
            text_input = "unknown"

        url = "https://api.openai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        params = {"model": model_type, "input": text_input}
        
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(url=url, headers=headers, json=params, timeout=BASE_TIMEOUT)
                if response.status_code == 200:
                    result_embeddings[key] = response.json()['data'][0]['embedding']
                    break
                elif response.status_code in [429, 500, 502, 503, 504]:
                    time.sleep(2 * (attempt + 1))
                else: 
                    break
            except Exception:
                time.sleep(2)
                continue
        if key not in result_embeddings:
            result_embeddings[key] = np.zeros(1536)
    return index, result_embeddings

# --- Main Steps ---

def step1(file_path, model_type, dataset):
    print(f"Step 1: Requesting LLM augmentation for {dataset}...")
    file_name = "augmented_attribute_dict"
    full_path = os.path.join(file_path, file_name)

    augmented_attribute_dict = {}

    # 📌 만약 이전에 저장해둔 파일이 있다면, 
    # 처음부터 다시 시작하지 않고 그 파일(딕셔너리)을 불러옵니다.
    if os.path.exists(full_path):
        with open(full_path, 'rb') as f:
            augmented_attribute_dict = pickle.load(f)

    # 데이터 로딩 및 초기 생성 (books의 경우를 위해 복구)
    attr_path = os.path.join(file_path, 'item_attribute.csv')
    
    if not os.path.exists(attr_path) and dataset.lower() in {"books", "yelp"}:
        print(f"❗ {attr_path} not found. Generating from JSON...")
        if dataset.lower() == "books":
            meta = pd.read_json("./data/books/item_meta_2017_kcore10_user_item_split_filtered.json", lines=True)
            df = meta[["item_id", "brand", "title", "category"]].copy()
        elif dataset.lower() == "yelp":
            meta = pd.read_json("./data/yelp/item_meta_2018_kcore5_user_item_split_filtered.json", lines=True)
            df = meta[["item_id", "name", "address", "city", "state", "categories", "stars", "review_count"]].copy()
        df = df.rename(columns={"item_id": "id"})
        df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
        df = df.sort_values("id").fillna("unknown")
        df.to_csv(attr_path, index=False, header=None)
        
    df = load_item_attribute(file_path, dataset)
    expected_count = len(df)
    print(f"Step 1 item_attribute rows: {expected_count}")
    print_item_coverage("Step 1 existing augmented_attribute_dict", expected_count, augmented_attribute_dict.keys())
    
    for pass_idx in range(STEP1_MAX_PASSES):
        target_indices = [i for i in range(expected_count) if i not in augmented_attribute_dict]
        print(f"Step 1 pass {pass_idx + 1}/{STEP1_MAX_PASSES}: processing {len(target_indices)} missing items...")
        if not target_indices:
            break

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(LLM_request_worker, idx, df, model_type, dataset): idx for idx in target_indices}
            for i, future in enumerate(tqdm(as_completed(futures), total=len(target_indices))):
                result = future.result()
                if result:
                    idx, data = result
                    print(f"✅ Augmented item {idx}: {data}")
                    augmented_attribute_dict[idx] = data
                if (i + 1) % 100 == 0:
                    with open(full_path, 'wb') as f: pickle.dump(augmented_attribute_dict, f)

        with open(full_path, 'wb') as f: pickle.dump(augmented_attribute_dict, f)
        print_item_coverage(f"Step 1 after pass {pass_idx + 1}", expected_count, augmented_attribute_dict.keys())

    with open(full_path, 'wb') as f: pickle.dump(augmented_attribute_dict, f)
    print_item_coverage("Step 1 final augmented_attribute_dict", expected_count, augmented_attribute_dict.keys())

def step2(file_path, dataset):
    print(f"Step 2: Aggregating LLM results for {dataset}...")
    attr_path = os.path.join(file_path, 'item_attribute.csv')
    
    cols_map = {
        "netflix": (['id', 'year', 'title'], ["director", "country", "language"]),
        "movielens": (['id', 'year', 'title', 'genre'], ["director", "country", "language"]),
        "books": (['id', 'brand', 'title', 'category'], ["author", "country", "language"]),
        "mind": (['id', 'title', 'category', 'subcategory'], ["publisher", "country", "language"]),
        "yelp": (
            ['id', 'name', 'address', 'city', 'state', 'categories', 'stars', 'review_count'],
            ["ambience_type", "visit_purpose", "companions_type"]
        )
    }
    
    base_cols, new_cols = cols_map[dataset.lower()]
    df = pd.read_csv(attr_path, names=base_cols)
    expected_count = len(df)
    print(f"Step 2 item_attribute rows: {expected_count}")

    with open(os.path.join(file_path, "augmented_attribute_dict"), "rb") as f:
        attr_dict = pickle.load(f)
    print_item_coverage("Step 2 source augmented_attribute_dict", expected_count, attr_dict.keys())

    for i, col_name in enumerate(new_cols):
        df[col_name] = [attr_dict.get(idx, {}).get(i, "unknown") for idx in range(len(df))]

    df.to_csv(os.path.join(file_path, "augmented_item_attribute_agg.csv"), index=False, header=None)
    print(f"Step 2 augmented_item_attribute_agg rows: {len(df)}")

def step3(file_path, emb_model, dataset):
    print("Step 3: Generating Embeddings...")
    agg_path = os.path.join(file_path, 'augmented_item_attribute_agg.csv')
    
    cols_map = {
        "netflix": ["id", "year", "title", "director", "country", "language"],
        "movielens": ["id", "year", "title", "genre", "director", "country", "language"],
        "books": ["id", "brand", "title", "category", "author", "country", "language"],
        "mind": ["id", "title", "category", "subcategory", "publisher", "country", "language"],
        "yelp": ["id", "name", "address", "city", "state", "categories", "stars", "review_count", "price_range", "service_options", "ambience"]
    }
    
    df = pd.read_csv(agg_path, names=cols_map[dataset.lower()])
    expected_count = len(df)
    print(f"Step 3 augmented_item_attribute_agg rows: {expected_count}")
    cols_to_emb = [c for c in df.columns if c != 'id']
    df[cols_to_emb] = df[cols_to_emb].fillna("unknown").astype(str)

    batch_size = 500
    for b_idx in range(0, len(df), batch_size):
        batch_num = (b_idx // batch_size) + 1
        out_name = f"augmented_attribute_embedding_dict{batch_num}"
        out_path = os.path.join(file_path, out_name)
        expected_indices = set(range(b_idx, min(b_idx + batch_size, len(df))))
        if os.path.exists(out_path):
            with open(out_path, 'rb') as f:
                existing = pickle.load(f)
            if all(expected_indices <= set(existing.get(col, {}).keys()) for col in cols_to_emb):
                print(f"Step 3 batch {batch_num}: already complete ({len(expected_indices)} items)")
                continue
            print(f"Step 3 batch {batch_num}: incomplete existing file, regenerating")

        batch_df = df.iloc[b_idx : b_idx + batch_size]
        res_dict = {col: {} for col in cols_to_emb}
        
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = [executor.submit(LLM_embedding_worker, idx, batch_df.loc[idx].to_dict(), cols_to_emb, emb_model) for idx in batch_df.index]
            for future in as_completed(futures):
                idx, embs = future.result()
                for col, v in embs.items(): res_dict[col][idx] = v

        with open(out_path, 'wb') as f: pickle.dump(res_dict, f)
        for col in cols_to_emb:
            print(f"Step 3 batch {batch_num} {col}: embedded_items={len(res_dict[col])}/{len(batch_df)}")

def step4(file_path, dataset):
    print("Step 4: Merging Batches...")
    cols_map = {
        "netflix": ['year', 'title', 'director', 'country', 'language'],
        "movielens": ['year', 'title', 'genre', 'director', 'country', 'language'],
        "books": ['brand', 'title', 'category', 'author', 'country', 'language'],
        "mind": ['title', 'category', 'subcategory', 'publisher', 'country', 'language'],
        "yelp": ['name', 'address', 'city', 'state', 'categories', 'stars', 'review_count', 'price_range', 'service_options', 'ambience']
    }
    
    keys = cols_map[dataset.lower()]
    total_dict = {k: {} for k in keys}
    expected_count = len(load_item_attribute(file_path, dataset))
    batch_size = 500
    expected_batches = (expected_count + batch_size - 1) // batch_size
    print(f"Step 4 expected item_attribute rows: {expected_count}")
    
    for i in range(1, expected_batches + 1):
        p = os.path.join(file_path, f"augmented_attribute_embedding_dict{i}")
        if not os.path.exists(p):
            print(f"⚠️ Step 4 missing batch file: {p}")
            continue
        with open(p, 'rb') as f:
            tmp = pickle.load(f)
            for k in keys:
                total_dict[k].update({
                    int(idx): emb
                    for idx, emb in tmp.get(k, {}).items()
                    if int(idx) < expected_count
                })

    with open(os.path.join(file_path, 'augmented_attribute_embedding_dict'), 'wb') as f:
        pickle.dump(total_dict, f)
    for k in keys:
        print_item_coverage(f"Step 4 merged {k}", expected_count, total_dict[k].keys())

def step5(file_path, dataset):
    print("Step 5: Creating Final Numpy Matrix...")
    with open(os.path.join(file_path, 'augmented_attribute_embedding_dict'), "rb") as f:
        agg_dict = pickle.load(f)

    try:
        with open(os.path.join(file_path, 'train_mat'), 'rb') as f:
            n_items = pickle.load(f).shape[1]
    except FileNotFoundError:
        n_items = max([max(d.keys()) for d in agg_dict.values() if d]) + 1
    expected_count = len(load_item_attribute(file_path, dataset))
    print(f"Step 5 expected item_attribute rows: {expected_count}, train_mat/items target: {n_items}")
    if n_items != expected_count:
        print(f"⚠️ Step 5 item count mismatch: train_mat/items target={n_items}, item_attribute rows={expected_count}")

    total_matrix = {}
    for key, val_dict in agg_dict.items():
        print_item_coverage(f"Step 5 source embedding {key}", expected_count, val_dict.keys())
        vecs = [val_dict.get(i, np.zeros(1536)) for i in range(n_items)]
        total_matrix[key] = np.array(vecs)
        print(f"Step 5 matrix {key}: shape={total_matrix[key].shape}")
    
    with open(os.path.join(file_path, 'augmented_total_embed_dict'), 'wb') as f:
        pickle.dump(total_matrix, f)

def main(dataset="yelp"):
    model_type = "gpt-4o" 
    emb_model = "text-embedding-3-small"
    
    dataset = dataset.lower()
    
    paths = {
        "netflix": "./LLMRec/LLMRec_c/netflix/netflix_valid_item",
        "movielens": "./data/ml-1m/ml-1m_llmrec_format/",
        "books": "./data/books/books_llmrec_format/",
        "mind": "./data/mind/mind_llmrec_format/",
        "yelp": "./data/yelp/yelp_llmrec_format/"
    }
    file_path = paths[dataset]

    os.makedirs(file_path, exist_ok=True) # 폴더가 없으면 자동 생성

    step1(file_path, model_type, dataset)
    step2(file_path, dataset)
    step3(file_path, emb_model, dataset)
    step4(file_path, dataset)
    step5(file_path, dataset)

if __name__ == '__main__':
    main("yelp")
