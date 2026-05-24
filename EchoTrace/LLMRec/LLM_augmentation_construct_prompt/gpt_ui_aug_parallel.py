import threading
import os
import time
import pickle
import requests
import pandas as pd
import numpy as np
import concurrent.futures
from tqdm import tqdm
import torch


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

# --- Prompt Construction (기존 로직 유지) ---
def construct_prompting(item_attribute, item_list, candidate_list, dataset):
    # 공통: 후보군 문자열 생성 최적화
    candidate_string = "Candidates:\n"
    for index in candidate_list:
        idx = index.item() if isinstance(index, (torch.Tensor, np.generic)) else int(index)
        
        if dataset.lower() == "netflix":
            candidate_string += f"[{idx}] {item_attribute['year'][idx]}, {item_attribute['title'][idx]}\n"
        elif dataset.lower() == "ml-1m":
            candidate_string += f"[{idx}] {item_attribute['year'][idx]}, {item_attribute['title'][idx]}, {item_attribute['genre'][idx]}\n"
        elif dataset.lower() == "books":
            candidate_string += f"[{idx}] {item_attribute['brand'][idx]}, {item_attribute['title'][idx]}, {item_attribute['category'][idx]}\n"
        elif dataset.lower() == "mind":
            candidate_string += f"[{idx}] {item_attribute['title'][idx]}, {item_attribute['category'][idx]}, {item_attribute['subcategory'][idx]}\n"
        elif dataset.lower() == "yelp":
            candidate_string += (
                f"[{idx}] {item_attribute['name'][idx]}, {item_attribute['city'][idx]}, "
                f"{item_attribute['state'][idx]}, {item_attribute['categories'][idx]}, "
                f"stars: {item_attribute['stars'][idx]}, review_count: {item_attribute['review_count'][idx]}\n"
            )

    # 데이터셋별 프롬프트 구성
    if dataset.lower() == "netflix" or dataset.lower() == "ml-1m":
        history_string = "User history:\n"
        for index in item_list:
            if dataset.lower() == "netflix":
                history_string += f"[{index}] {item_attribute['year'][index]}, {item_attribute['title'][index]}\n"
            else:
                history_string += f"[{index}] {item_attribute['year'][index]}, {item_attribute['title'][index]}, {item_attribute['genre'][index]}\n"

        output_format = (
            "Please output the index of user's favorite and least favorite movie only from candidate, but not user history.\n"
            "Output format:\nTwo numbers separated by '::'. Nothing else.\n"
            "Please just give the index of candidates, remove [], do not output other things, no reasoning.\n\n"
        )
        prompt = (
            "You are a movie recommendation system and required to recommend user with movies based on user history that each movie with title (same topic/doctor), year (similar years), genre (similar genre).\n"
            + history_string + candidate_string + output_format
        )

    elif dataset.lower() == "books":
        history_string = "User history:\n"
        for index in item_list:
            history_string += f"[{index}] {item_attribute['brand'][index]}, {item_attribute['title'][index]}, {item_attribute['category'][index]}\n"

        output_format = (
            "Please output the index of user's favorite and least favorite book only from candidate, but not user history.\n"
            "Output format:\nTwo numbers separated by '::'. Nothing else.\n"
            "Please just give the index of candidates, remove [], do not output other things, no reasoning.\n\n"
        )   
        prompt = (
            "You are a book recommendation system and required to recommend user with books based on user history that each book with brand, title, category.\n"
            + history_string + candidate_string + output_format
        )
        
    elif dataset.lower() == "mind":
        history_string = "User history:\n"
        for index in item_list:
            history_string += f"[{index}] {item_attribute['title'][index]}, {item_attribute['category'][index]}, {item_attribute['subcategory'][index]}\n"

        output_format = (
            "Please output the index of user's favorite and least favorite news article only from candidate, but not user history.\n"
            "Output format:\nTwo numbers separated by '::'. Nothing else.\n"
            "Please just give the index of candidates, remove [], do not output other things, no reasoning.\n\n"
        )   
        prompt = (
            "You are a news recommendation system and required to recommend user with news articles based on user history that each news with title, category, subcategory.\n"
            + history_string + candidate_string + output_format
        )

    elif dataset.lower() == "yelp":
        history_string = "User history:\n"
        for index in item_list:
            history_string += (
                f"[{index}] {item_attribute['name'][index]}, {item_attribute['city'][index]}, "
                f"{item_attribute['state'][index]}, {item_attribute['categories'][index]}, "
                f"stars: {item_attribute['stars'][index]}, review_count: {item_attribute['review_count'][index]}\n"
            )

        output_format = (
            "Please output the index of user's favorite and least favorite Yelp business only from candidate, but not user history.\n"
            "Output format:\nTwo numbers separated by '::'. Nothing else.\n"
            "Please just give the index of candidates, remove [], do not output other things, no reasoning.\n\n"
        )
        prompt = (
            "You are a Yelp business recommendation system and required to recommend businesses based on user history "
            "that each business has name, location, categories, stars, and review_count.\n"
            + history_string + candidate_string + output_format
        )
    
    return prompt

def _require_key(key: str, name: str):
    if not key:
        raise RuntimeError(f"{name} is not set in environment variables.")
    return key


def get_system_message(dataset):
    if dataset.lower() == "books":
        return "You are a book recommendation system."
    elif dataset.lower() == "mind":
        return "You are a news recommendation system."
    elif dataset.lower() == "yelp":
        return "You are a Yelp business recommendation system."
    else:
        return "You are a movie recommendation system."


def call_llm_ui(provider: str, model: str, prompt: str, sys_msg: str, timeout: int = 20) -> str:
    provider = provider.lower().strip()
    if provider == "openai":
        return _call_openai_chat(model=model, prompt=prompt, sys_msg=sys_msg, timeout=timeout)
    if provider == "anthropic":
        return _call_anthropic_messages(model=model, prompt=prompt, sys_msg=sys_msg, timeout=timeout)
    raise ValueError(f"Unknown provider: {provider} (use 'openai' or 'anthropic').")


def _call_openai_chat(model: str, prompt: str, sys_msg: str, timeout: int = 20) -> str:
    key = _require_key(OPENAI_API_KEY, "OPENAI_API_KEY")
    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + key,
    }
    params = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_msg}, 
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1024, # 응답이 짧으므로 줄임
        "temperature": 0.7,
        "stream": False
    }
    response = requests.post(url=url, headers=headers, json=params, timeout=timeout)
    if response.status_code != 200:
        raise requests.exceptions.RequestException(f"OpenAI API Error {response.status_code}: {response.text}")
    message = response.json()
    return message["choices"][0]["message"]["content"]


def _call_anthropic_messages(model: str, prompt: str, sys_msg: str, timeout: int = 20) -> str:
    key = _require_key(ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY")
    url = f"{ANTHROPIC_BASE_URL}/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
    }
    params = {
        "model": model,
        "max_tokens": 1024,
        "temperature": 0.7,
        "system": sys_msg,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    response = requests.post(url=url, headers=headers, json=params, timeout=timeout)
    if response.status_code != 200:
        raise requests.exceptions.RequestException(f"Anthropic API Error {response.status_code}: {response.text}")
    message = response.json()

    texts = []
    for block in message.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            texts.append(block.get("text", ""))
    return "".join(texts).strip()


def parse_recommendation_response(content):
    samples = content.strip().split("::")
    if len(samples) < 2:
        # 가끔 설명이 붙는 경우 처리 시도 (숫자만 추출)
        import re
        nums = re.findall(r'\d+', content)
        if len(nums) >= 2:
            return int(nums[0]), int(nums[1])
        raise ValueError("Parsing Failed")
    return int(samples[0].strip()), int(samples[1].strip())


# --- Worker Function for Threading ---
def LLM_request_worker(args):
    index, toy_item_attribute, adjacency_list, candidate_list, provider, model_type, dataset = args

    prompt = construct_prompting(toy_item_attribute, adjacency_list, candidate_list, dataset)
    sys_msg = get_system_message(dataset)

    for retry in range(3): # 재시도 횟수 3회로 조정
        try:
            content = call_llm_ui(
                provider=provider,
                model=model_type,
                prompt=prompt,
                sys_msg=sys_msg,
                timeout=20,
            )
            pos_sample, neg_sample = parse_recommendation_response(content)

            # print(f"✅ Index {index} -> Pos: {pos_sample}, Neg: {neg_sample}")
            return index, {0: pos_sample, 1: neg_sample}

        except Exception as e:
            # print(f"❌ Error at index {index}: {e}")
            time.sleep(2)

    return index, None # 실패 시 None 반환

def main(dataset="books", file_path=None, provider="anthropic"):
    provider = provider.lower().strip()
    if provider not in {"openai", "anthropic"}:
        raise ValueError(f"Unknown provider: {provider} (use 'openai' or 'anthropic').")

    # 경로 설정
    if dataset == "netflix":
        default_file_path = "./LLMRec/LLMRec_c/" + dataset + "/netflix_valid_item/"
        if file_path is None:
            file_path = default_file_path
        toy_item_attribute = pd.read_csv(os.path.join(file_path, 'item_attribute.csv'), names=['id', 'year', 'title'])
    elif dataset == "ml-1m":
        default_file_path = "./data/ml-1m/ml-1m_llmrec_format/"
        if file_path is None:
            file_path = default_file_path
        toy_item_attribute = pd.read_csv(os.path.join(file_path, 'item_attribute.csv'), names=['id', 'year', 'title', 'genre'])
    elif dataset == "books":
        default_file_path = "./data/books/books_llmrec_format/"
        if file_path is None:
            file_path = default_file_path
        toy_item_attribute = pd.read_csv(os.path.join(file_path, 'item_attribute.csv'), names=['id', 'brand', 'title', 'category'])
    elif dataset == "mind":
        default_file_path = "./data/mind/mind_llmrec_format/"
        if file_path is None:
            file_path = default_file_path
        toy_item_attribute = pd.read_csv(os.path.join(file_path, 'item_attribute.csv'), names=['id', 'category', 'subcategory', 'title'])
    elif dataset == "yelp":
        default_file_path = "./data/yelp/yelp_llmrec_format/"
        if file_path is None:
            file_path = default_file_path
        toy_item_attribute = pd.read_csv(
            os.path.join(file_path, 'item_attribute.csv'),
            names=['id', 'name', 'address', 'city', 'state', 'categories', 'stars', 'review_count']
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset}")

    gen_model_openai = "gpt-4o"
    gen_model_anthropic = "claude-haiku-4-5"
    model_type = gen_model_openai if provider == "openai" else gen_model_anthropic
    aug_path = os.path.join(file_path, "augmented_sample_dict")

    # 1. 데이터 로드
    print("📂 Loading Data...")
    candidate_indices = pickle.load(open(os.path.join(file_path, 'candidate_indices'), 'rb'))
    candidate_indices_dict = {i: candidate_indices[i] for i in range(candidate_indices.shape[0])}

    train_mat = pickle.load(open(os.path.join(file_path, 'train_mat'),'rb'))
    adjacency_list_dict = {}
    for index in range(train_mat.shape[0]):
        _, data_y = train_mat[index].nonzero()
        adjacency_list_dict[index] = data_y
        
    # 2. 증강 결과 새로 생성
    augmented_sample_dict = {}
    print("🆕 Starting new dictionary.")

    # 3. 작업 대상 선정 (이미 완료된 인덱스 제외)
    all_indices = list(adjacency_list_dict.keys())
    target_indices = [i for i in all_indices if i not in augmented_sample_dict]
    
    print(f"🚀 Processing {len(target_indices)} users with Multithreading...")

    failed_indices = []
    max_workers = 10  # 스레드 수 설정
    save_interval = 20 # 저장 간격
    batch_cnt = 0

    # 4. 병렬 처리 실행
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 작업 큐 생성
        futures = {
            executor.submit(
                LLM_request_worker, 
                (
                    idx,
                    toy_item_attribute,
                    adjacency_list_dict[idx][-10:],
                    candidate_indices_dict[idx],
                    provider,
                    model_type,
                    dataset,
                )
            ): idx for idx in target_indices
        }

        # 결과 수집 (tqdm으로 진행률 표시)
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(target_indices), desc="Augmenting"):
            idx, result = future.result()
            
            if result is not None:
                augmented_sample_dict[idx] = result
                batch_cnt += 1
            else:
                failed_indices.append(idx)

            # 배치 저장
            if batch_cnt >= save_interval:
                with open(aug_path, 'wb') as f:
                    pickle.dump(augmented_sample_dict, f)
                batch_cnt = 0
                # print("💾 Progress saved.")

    # 최종 저장
    with open(aug_path, 'wb') as f:
        pickle.dump(augmented_sample_dict, f)
    
    # 실패 목록 저장
    if failed_indices:
        with open(os.path.join(file_path, 'failed_ui_aug_indices.pkl'), 'wb') as f:
            pickle.dump(failed_indices, f)
        print(f"❗ {len(failed_indices)} indices failed.")
    else:
        print("✅ All processed successfully.")

if __name__ == '__main__':
    # 실행할 데이터셋 선택 (netflix, ml-1m, books, yelp)
    main("yelp")
