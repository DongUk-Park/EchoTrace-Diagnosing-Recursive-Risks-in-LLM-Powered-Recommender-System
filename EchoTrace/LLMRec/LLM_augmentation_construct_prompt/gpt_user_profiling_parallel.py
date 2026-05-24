import threading
import time
import pandas as pd
import csv
import requests
import concurrent.futures
import pickle
import torch
import os
import time
import numpy as np
import json
import random

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

def construct_user_prompt(item_attribute, item_list, dataset):
    if dataset.lower() == "netflix":
        history_string = "User history:\n"
        for index in item_list:
            year = item_attribute['year'][index]
            title = item_attribute['title'][index]
            history_string += f"[{index}] {year}, {title}\n"

        output_format = (
            "Please output the following infomation of user, output json format(json):\n"
            "{'age': 'predicted age', 'gender':'predicted gender', 'liked genre':'predicted liked genre', 'disliked genre':'predicted disliked genre', \
            'liked directors':'predicted liked directors', 'country':'predicted country', 'language':'predicted language'}\n"
            "In json format, don't forget to wrap the prediction data corresponding to the right side with ' '.\n"
            "Please do not fill in 'unknown', but make an educated guess.\n"
            "Only output the content after \"output format:\", no reasoning or other content.\n\n"
        )
        prompt = (
            "You are required to generate user profile based on the history of user, \
            that each movie with title, year.\n"
            + history_string + output_format
        )
    elif dataset.lower() == "ml-1m":
        history_string = "User history:\n"
        for index in item_list[-10:]:
            title = item_attribute['title'][index]
            year = item_attribute['year'][index]
            genre = item_attribute['genre'][index]
            history_string += f"([{index}] {year}, {title}, {genre})\n"

        output_format = (
            "Please output the following infomation of user, output format(json):\n"
            "{'age': 'predicted age', 'gender': 'predicted gender', 'occupation': 'predicted occupation', 'liked_genre':'predicted liked genre', 'disliked_genre':'predicted disliked genre', \
            'liked_directors':'predicted liked directors', 'country':'predicted country', 'language':'predicted language'}\n"
        )
        prompt = (
            "You are User Analyst. Based on the user history below (each entry contains title, year, and genre), \
            predict the following: age, gender, occupation, liked_genre, disliked_genre, liked_directors, country, language.\n"
            "Output only the following JSON (wrap all values with single quotes):, \n\n" + history_string + output_format
        )
        
    elif dataset.lower() == "books":
        history_string = "User history:\n"
        for index in item_list[-10:]:
            brand = item_attribute['brand'][index]
            title = item_attribute['title'][index]
            category = item_attribute['category'][index]
            history_string += f"([{index}] {brand}, {title}, {category})\n"

        output_format = (
            "Please output the following infomation of user, output format(json):\n"
            "{'age': 'predicted age', 'gender':'predicted gender', 'liked category':'predicted liked category', 'disliked category':'predicted disliked category', \
            'liked author':'predicted liked author', 'country':'predicted country', 'language':'predicted language'}\n"
            "In json format, don't forget to wrap the prediction data corresponding to the right side with ' '.\n"
            "Please do not fill in 'unknown', but make an educated guess.\n"
            "Only output the content after \"output format:\", no reasoning or other content.\n\n"
        )
        prompt = (
            "You are a Book Recommendation Specialist. "
            "You are required to generate user profile based on the history of user, \
            that each entry contains brand, title, and category.\n" + history_string + output_format
        )

    elif dataset.lower() == "yelp":
        history_string = "User visit history:\n"
        for index in item_list[-10:]:
            name = item_attribute["name"][index]
            city = item_attribute["city"][index]
            state = item_attribute["state"][index]
            categories = item_attribute["categories"][index]
            stars = item_attribute["stars"][index]
            review_count = item_attribute["review_count"][index]
            history_string += f"([{index}] {name}, {city}, {state}, {categories}, stars: {stars}, review_count: {review_count})\n"

        output_format = (
            "Please output the following information of user, output format(json):\n"
            "{'age':'predicted age', 'gender':'predicted gender', "
            "'occupation':'predicted occupation', "
            "'liked_business_category':'predicted liked business category', "
            "'disliked_business_category':'predicted disliked business category', "
            "'preferred_city':'predicted preferred city'}\n"
            "In json format, wrap all values with single quotes.\n"
            "Please do not fill in 'unknown', but make an educated guess based on visit behavior.\n"
            "Only output JSON, no reasoning or explanation.\n\n"
        )

        prompt = (
            "You are a POI Recommendation Analyst. "
            "Based on the user's business visit history below (each entry contains name, location, categories, stars, and review_count), "
            "infer demographic and preference profile of the user.\n\n"
            + history_string + output_format
        )

    else:
        raise ValueError(f"Unknown dataset type: {dataset}")

    return prompt

def _require_key(key: str, name: str):
    if not key:
        raise RuntimeError(f"{name} is not set in environment variables.")
    return key

def call_llm_profile(provider: str, model: str, prompt: str, temp: float, timeout: int = 30) -> str:
    provider = provider.lower().strip()
    if provider == "openai":
        return _call_openai_chat(model=model, prompt=prompt, temp=temp, timeout=timeout)
    elif provider == "anthropic":
        return _call_anthropic_messages(model=model, prompt=prompt, temp=temp, timeout=timeout)
    else:
        raise ValueError(f"Unknown provider: {provider} (use 'openai' or 'anthropic').")

def _call_openai_chat(model: str, prompt: str, temp: float, timeout: int = 30) -> str:
    key = _require_key(OPENAI_API_KEY, "OPENAI_API_KEY")
    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + key,
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp,
        "top_p": 0.1,
        "stream": False,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if r.status_code != 200:
        raise requests.exceptions.RequestException(f"OpenAI API Error {r.status_code}: {r.text}")
    j = r.json()
    return j["choices"][0]["message"]["content"]

def _call_anthropic_messages(model: str, prompt: str, temp: float, timeout: int = 30) -> str:
    key = _require_key(ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY")
    url = f"{ANTHROPIC_BASE_URL}/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": model,
        "max_tokens": 1024,
        "temperature": temp,
        #"top_p": 0.1,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if r.status_code != 200:
        raise requests.exceptions.RequestException(f"Anthropic API Error {r.status_code}: {r.text}")
    j = r.json()

    # Anthropic: content = [{"type":"text","text":"..."}, ...]
    blocks = j.get("content", [])
    texts = []
    for b in blocks:
        if isinstance(b, dict) and b.get("type") == "text":
            texts.append(b.get("text", ""))
    return "".join(texts).strip()

def LLM_request(toy_item_attribute, adjacency_list_dict, index, provider, model_type,
                augmented_user_profiling_dict, error_cnt, dataset, temp):
    if index in augmented_user_profiling_dict:
        return index, None

    try:
        prompt = construct_user_prompt(toy_item_attribute, adjacency_list_dict[index], dataset)
        content = call_llm_profile(provider=provider, model=model_type, prompt=prompt, temp=temp, timeout=30)
        print(f"\n[DEBUG] User Index: {index} Profile Generated:")
        print(content)
        print("-" * 50)
        return index, content

    except Exception as ex:
        print(f"Error at index {index}: {ex}")
        error_cnt += 1
        if error_cnt >= 5:
            return index, None
        time.sleep(3)
        return LLM_request(toy_item_attribute, adjacency_list_dict, index, provider, model_type,
                           augmented_user_profiling_dict, error_cnt, dataset, temp)

def LLM_embedding_request(
    augmented_user_profiling_dict,
    index,
    model_type,
    augmented_user_init_embedding,
    max_retries=6,
):
    # step2는 OpenAI embedding 그대로 유지 권장 (Anthropic은 embedding 모델 없음)
    if index in augmented_user_init_embedding:
        return index, None

    key = _require_key(OPENAI_API_KEY, "OPENAI_API_KEY")
    url = f"{OPENAI_BASE_URL}/embeddings"
    headers = {"Authorization": "Bearer " + key}
    params = {"model": model_type, "input": augmented_user_profiling_dict[index]}

    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=params, timeout=30)
            if r.status_code != 200:
                raise requests.exceptions.RequestException(
                    f"OpenAI Embedding Error {r.status_code}: {r.text}"
                )
            j = r.json()
            emb = j["data"][0]["embedding"]
            return index, np.array(emb, dtype=np.float32)

        except Exception as ex:
            msg = str(ex).lower()
            retryable = (
                "429" in msg
                or "500" in msg
                or "502" in msg
                or "503" in msg
                or "504" in msg
                or "bad gateway" in msg
                or "timeout" in msg
                or "connection" in msg
            )

            if not retryable or attempt == max_retries - 1:
                print(f"Embedding failed at index {index}: {ex}")
                return index, None

            sleep_sec = min(60, (2 ** attempt) + random.uniform(0, 1))
            print(
                f"Embedding retry at index {index} "
                f"({attempt + 1}/{max_retries}) after {sleep_sec:.1f}s: {ex}"
            )
            time.sleep(sleep_sec)

def step1(file_path, provider, g_model_type, start_id, dataset, temp):
    augmented_user_profiling_dict = {}
    dict_path = os.path.join(file_path, f"augmented_user_profiling_dict")
    with open(dict_path, "wb") as f:
        pickle.dump(augmented_user_profiling_dict, f)

    if dataset.lower() == "netflix":
        toy_item_attribute = pd.read_csv(os.path.join(file_path, "item_attribute.csv"), names=["id", "year", "title"])
    elif dataset.lower() == "ml-1m":
        toy_item_attribute = pd.read_csv(os.path.join(file_path, "item_attribute.csv"), names=["id", "year", "title", "genre"])
    elif dataset.lower() == "books":
        toy_item_attribute = pd.read_csv(os.path.join(file_path, "item_attribute.csv"), names=["id", "brand", "title", "category"])
    elif dataset.lower() == "yelp":
        toy_item_attribute = pd.read_csv(
            os.path.join(file_path, "item_attribute.csv"),
            names=["id", "name", "address", "city", "state", "categories", "stars", "review_count"]
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset}")

    train_mat = pickle.load(open(os.path.join(file_path, "train_mat"), "rb"))
    adjacency_list_dict = {}
    for u in range(train_mat.shape[0]):
        _, data_y = train_mat[u].nonzero()
        adjacency_list_dict[u] = data_y

    target_indices = list(range(start_id, len(adjacency_list_dict)))
    max_workers = 10
    batch_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                LLM_request,
                toy_item_attribute,
                adjacency_list_dict,
                idx,
                provider,
                g_model_type,
                augmented_user_profiling_dict,
                0,
                dataset,
                temp
            ): idx for idx in target_indices
        }

        for future in concurrent.futures.as_completed(futures):
            idx, content = future.result()
            if content is not None:
                augmented_user_profiling_dict[idx] = content
                batch_count += 1

            if batch_count > 0 and batch_count % 20 == 0:
                with open(dict_path, "wb") as f:
                    pickle.dump(augmented_user_profiling_dict, f)
                batch_count = 0

    with open(dict_path, "wb") as f:
        pickle.dump(augmented_user_profiling_dict, f)

def step2(emb_model, file_path, profile_step_num=None, parts=None):
    if profile_step_num is None:
        dict_path = os.path.join(file_path, "augmented_user_profiling_dict")
    else:
        if parts is None:
            raise ValueError("parts must be provided when profile_step_num is provided.")
        dict_path = os.path.join(
            file_path,
            f"augmented_user_profiling_dict_part{parts}_step{profile_step_num}",
        )

    if not os.path.exists(dict_path):
        raise FileNotFoundError(f"User profiling dict not found: {dict_path}")

    emb_path = os.path.join(file_path, "augmented_user_init_embedding")
    failed_path = os.path.join(file_path, "augmented_user_init_embedding_failed")

    print(f"Loading user profiling dict: {dict_path}")
    augmented_user_profiling_dict = pickle.load(open(dict_path, "rb"))
    augmented_user_init_embedding = {}
    with open(emb_path, "wb") as f:
        pickle.dump(augmented_user_init_embedding, f)

    failed_users = set()
    target_users = [
        user_id
        for user_id, profile in augmented_user_profiling_dict.items()
        if user_id not in augmented_user_init_embedding and profile
    ]
    max_workers = int(os.environ.get("EMBEDDING_MAX_WORKERS", "5"))
    batch_count = 0
    processed_count = 0

    print(
        f"Embedding step2 start: total_profiles={len(augmented_user_profiling_dict)}, "
        f"remaining={len(target_users)}, max_workers={max_workers}"
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                LLM_embedding_request,
                augmented_user_profiling_dict,
                user_id,
                emb_model,
                augmented_user_init_embedding
            ): user_id for user_id in target_users
        }

        for future in concurrent.futures.as_completed(futures):
            user_id = futures[future]
            try:
                idx, embedding = future.result()
            except Exception as ex:
                print(f"Embedding future failed at index {user_id}: {ex}")
                failed_users.add(user_id)
                continue

            processed_count += 1
            if embedding is not None:
                augmented_user_init_embedding[idx] = embedding
                batch_count += 1
            else:
                failed_users.add(idx)

            if batch_count > 0 and batch_count % 20 == 0:
                with open(emb_path, "wb") as f:
                    pickle.dump(augmented_user_init_embedding, f)
                with open(failed_path, "wb") as f:
                    pickle.dump(sorted(failed_users), f)
                print(
                    f"Saved embeddings: {len(augmented_user_init_embedding)} "
                    f"(processed {processed_count}/{len(target_users)}, failed {len(failed_users)})"
                )
                batch_count = 0

    with open(emb_path, "wb") as f:
        pickle.dump(augmented_user_init_embedding, f)
    with open(failed_path, "wb") as f:
        pickle.dump(sorted(failed_users), f)
    print(
        f"Embedding step2 done: saved={len(augmented_user_init_embedding)}, "
        f"failed={len(failed_users)}"
    )

def step3(file_path):
    embed_dict = pickle.load(open(os.path.join(file_path, "augmented_user_init_embedding"), "rb"))
    train_mat = pickle.load(open(os.path.join(file_path, "train_mat"), "rb"))
    n_users = train_mat.shape[0]

    final_matrix = []
    dim = 1536
    for i in range(n_users):
        if i in embed_dict:
            final_matrix.append(embed_dict[i])
        else:
            final_matrix.append(np.zeros(dim, dtype=np.float32))

    final_array = np.array(final_matrix)
    with open(os.path.join(file_path, "augmented_user_init_embedding_final"), "wb") as f:
        pickle.dump(final_array, f)

def main(dataset, provider="openai", file_path=None, profile_step_num=None, parts=None):
    if dataset == "netflix":
        default_file_path = "./LLMRec/LLMRec_c/" + dataset + "/netflix_valid_item/"
        gen_model_openai = "gpt-4o"
        gen_model_anthropic = "claude-haiku-4-5"
    elif dataset == "ml-1m":
        default_file_path = f"./data/{dataset}/{dataset}_llmrec_format/"
        gen_model_openai = "gpt-4o"
        gen_model_anthropic = "claude-haiku-4-5"
    elif dataset == "books":
        default_file_path = f"./data/{dataset}/{dataset}_llmrec_format/"
        gen_model_openai = "gpt-4"
        gen_model_anthropic = "claude-haiku-4-5"
    elif dataset == "yelp":
        default_file_path = f"./data/{dataset}/{dataset}_llmrec_format/"
        gen_model_openai = "gpt-4o"
        gen_model_anthropic = "claude-haiku-4-5"
    else:
        raise ValueError(dataset)

    if file_path is None:
        file_path = default_file_path

    start_id = 0
    embedding_model = "text-embedding-3-small"

    gen_model = gen_model_openai if provider == "openai" else gen_model_anthropic

    if profile_step_num is None and parts is None:
        print("Starting full profiling and embedding generation...")
        step1(file_path, provider, gen_model, start_id, dataset, temp=0.6)
    step2(embedding_model, file_path, profile_step_num=profile_step_num, parts=parts)
    step3(file_path)

if __name__ == "__main__":
    main("yelp", provider="openai")  # "openai", "anthropic"
