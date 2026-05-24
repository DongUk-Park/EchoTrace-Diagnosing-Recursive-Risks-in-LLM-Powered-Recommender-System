import gzip
import pickle
from collections import defaultdict
import os
import gzip
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer, OPTForCausalLM, AutoModelForCausalLM
import torch
import pandas as pd


def preprocess_movielens_1m(data_dir='./data/ml-1m', output_dir='./data/ml-1m', fname='ml-1m'):
    user_history = defaultdict(list)
    item_title_dict = {}
    item_description_dict = {}

    # Genre list (from u.genre)
    genre_list = []
    with open(f'{data_dir}/u.genre', encoding='ISO-8859-1') as f:
        for line in f:
            if line.strip():
                genre, _ = line.strip().split('|')
                genre_list.append(genre)

    # Read item titles and descriptions
    with open(f'{data_dir}/u.item', encoding='ISO-8859-1') as f:
        for line in f:
            parts = line.strip().split('|')
            item_id = int(parts[0])
            title = parts[1]
            genres_bin = list(map(int, parts[5:]))  # genre one-hot starts at index 5

            # Extract year from title if present
            if '(' in title and ')' in title[-6:]:
                year = title.strip()[-5:-1]
            else:
                year = "Unknown"

            genres = [g for g, flag in zip(genre_list, genres_bin) if flag]
            genre_str = ', '.join(genres) if genres else 'Unknown'

            item_title_dict[item_id] = title
            item_description_dict[item_id] = f"Released in {year}. Genres: {genre_str}"

    # Read user-item interactions
    with open(f'{data_dir}/u.data') as f:
        for line in f:
            user_id, item_id, rating, timestamp = map(int, line.strip().split('\t'))
            user_history[user_id].append((timestamp, item_id))
    
    for uid in user_history:
        user_history[uid].sort(key=lambda x: x[0])

    name_dict = {
        'title': item_title_dict,
        'description': item_description_dict
    }


    with gzip.open(f'{output_dir}/{fname}_text_name_dict.json.gz', 'wb') as tf:
        pickle.dump(name_dict, tf)

    with open(f'{output_dir}/{fname}.txt', 'w') as f:
        for uid in sorted(user_history.keys()):
            for _, iid in user_history[uid]:
                f.write(f"{uid} {iid}\n")

    print(f"✔️ Preprocessing complete: {len(user_history)} users, {len(item_title_dict)} items.")

def create_name_dict_from_OMDB(
    desc_path="./data/ml-1m/A-LLMRec_format/movie_descriptions_final.csv",
    output_path="./data/ml-1m/A-LLMRec_format/ml-1m_text_name_dict.json.gz",
):
    df = pd.read_csv(desc_path)

    # 컬럼 체크
    required = {"movie_id", "title", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {desc_path}: {sorted(missing)}")

    # 타입/결측 정리
    df["movie_id"] = pd.to_numeric(df["movie_id"], errors="raise").astype(int)
    df["title"] = df["title"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)

    df["title"] = df["title"].str.strip()
    df["description"] = df["description"].str.strip()

    # 빈 값도 None 처리
    df.loc[df["description"] == "", "description"] = "None"

    # 중복 movie_id 방지 (있으면 마지막 값 유지)
    df = df.drop_duplicates(subset=["movie_id"], keep="last")

    item_title_dict = dict(zip(df["movie_id"], df["title"]))
    item_description_dict = dict(zip(df["movie_id"], df["description"]))

    name_dict = {"title": item_title_dict, "description": item_description_dict}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with gzip.open(output_path, "wb") as f:
        pickle.dump(name_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✔️ name_dict saved to: {output_path} (items: {len(item_title_dict)})")
    return name_dict


def preprocess_movielens_100k_with_llm(
    data_dir='./A-LLMRec/data/ml-100k',
    output_dir='./A-LLMRec/data/amazon',
    fname='ml-100k',
    llm_model='opt',
    max_desc_length=30,
    device='cuda'
):
    os.makedirs(output_dir, exist_ok=True)

    if llm_model == 'opt':
        model_path = "facebook/opt-6.7b"
    elif llm_model == 'llama':
        model_path = "meta-llama/Meta-Llama-3-8B"
    else:
        raise Exception(f'{llm_model} is not supported')

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        use_safetensors=True
    ).eval()

    tokenizer.add_special_tokens({
        'pad_token': '[PAD]',
        'bos_token': '</s>',
        'eos_token': '</s>',
        'unk_token': '</s>',
        'additional_special_tokens': ['[UserRep]', '[HistoryEmb]', '[CandidateEmb]']
    })
    model.resize_token_embeddings(len(tokenizer))

    # Load genres
    genre_list = []
    with open(os.path.join(data_dir, 'u.genre'), encoding='ISO-8859-1') as f:
        for line in f:
            if line.strip():
                genre, _ = line.strip().split('|')
                genre_list.append(genre)

    # Load item metadata
    item_title_dict = {}
    item_desc_dict = {}
    with open(os.path.join(data_dir, 'u.item'), encoding='ISO-8859-1') as f:
        for line in tqdm(f, desc="Generating descriptions"):
            parts = line.strip().split('|')
            item_id = int(parts[0])
            title = parts[1]
            genre_flags = list(map(int, parts[5:]))
            genres = [g for g, f in zip(genre_list, genre_flags) if f]
            genre_str = ', '.join(genres) if genres else 'Unknown'

            # prompt 구성 (지침 제거 포함)
            prompt = f'This is a brief plot summary for the movie "{title}", which belongs to the {genre_str} genre. Plot:'

            inputs = tokenizer(prompt, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()} 

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=True,
                    temperature=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = generated.replace(prompt, '').strip()


            item_title_dict[item_id] = title
            item_desc_dict[item_id] = generated

    # Save name_dict
    name_dict = {
        'title': item_title_dict,
        'description': item_desc_dict
    }
    out_path = f'{output_dir}/{fname}_text_name_dict.json.gz'
    with gzip.open(out_path, 'wb') as f:
        pickle.dump(name_dict, f)

    print(f'✅ Saved text_name_dict to: {out_path}')
    return name_dict


# 사용 예시
if __name__ == "__main__":
    #preprocess_movielens_1m()
    create_name_dict_from_OMDB()
