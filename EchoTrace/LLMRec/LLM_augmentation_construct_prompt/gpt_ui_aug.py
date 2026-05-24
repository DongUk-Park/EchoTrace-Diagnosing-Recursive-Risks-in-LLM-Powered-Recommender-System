import threading
import openai
import time
import pandas as pd
import csv
import concurrent.futures
import pickle
import torch
import os
import time
import tqdm
import requests
API_KEY = os.environ.get("OPENAI_API_KEY")  # 없으면 None

# Netflix
def construct_prompting(item_attribute, item_list, candidate_list, dataset):
    if dataset.lower() == "netflix":
        history_string = "User history:\n"
        for index in item_list:
            year = item_attribute['year'][index]
            title = item_attribute['title'][index]
            history_string += f"[{index}] {year}, {title}\n"

        candidate_string = "Candidates:\n"
        for index in candidate_list:
            idx = index.item()
            year = item_attribute['year'][idx]
            title = item_attribute['title'][idx]
            candidate_string += f"[{idx}] {year}, {title}\n"
        output_format = (
            "Please output the index of user's favorite and least favorite movie only from candidate, but not user history.\n"
            "Output format:\nTwo numbers separated by '::'. Nothing else.\n"
            "Please just give the index of candidates, remove [], do not output other things, no reasoning.\n\n"
        )
        
        prompt = (
            "You are a movie recommendation system and required to recommend user with movies based on user history that each movie with title (same topic/doctor), year (similar years), genre (similar genre).\n"
            + history_string + candidate_string + output_format
        )
    elif dataset.lower() == "movielens":
        history_string = "User history:\n"
        for index in item_list:
            title = item_attribute['title'][index]
            year = item_attribute['year'][index]
            genre = item_attribute['genre'][index]
            history_string += f"[{index}] {year}, {title}, {genre}\n"

        candidate_string = "Candidates:\n"
        for index in candidate_list:
            idx = int(index)
            title = item_attribute['title'][idx]
            year = item_attribute['year'][idx]
            genre = item_attribute['genre'][idx]
            candidate_string += f"[{idx}] {year}, {title}, {genre}\n"

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
            brand = item_attribute['brand'][index]
            title = item_attribute['title'][index]
            category = item_attribute['category'][index]
            history_string += f"[{index}] {brand}, {title}, {category}\n"

        candidate_string = "Candidates:\n"
        for index in candidate_list:
            idx = index.item()
            brand = item_attribute['brand'][idx]
            title = item_attribute['title'][idx]
            category = item_attribute['category'][idx]
            candidate_string += f"[{idx}] {brand}, {title}, {category}\n"
        output_format = (
        "Please output the index of user's favorite and least favorite book only from candidate, but not user history.\n"
        "Output format:\nTwo numbers separated by '::'. Nothing else.\n"
        "Please just give the index of candidates, remove [], do not output other things, no reasoning.\n\n"
        )   
        prompt = (
            "You are a book recommendation system and required to recommend user with books based on user history that each book with brand, title, category.\n"
            + history_string + candidate_string + output_format
        )
    
    
    return prompt

# chatgpt
def LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, model_type, augmented_sample_dict, file_path, failed_indices, dataset):
    if index in augmented_sample_dict:
        print(f"✅ Already processed: {index}")
        return 0

    prompt = construct_prompting(toy_item_attribute, adjacency_list_dict[index], candidate_indices_dict[index], dataset)
    if dataset.lower() == "books":
        sys_msg = "You are a book recommendation system and required to recommend user with books based on user history that each book with brand, title, category.\n"
    elif dataset.lower() in ["netflix", "movielens"]:
        sys_msg = "You are a movie recommendation system and required to recommend user with movies based on user history that each movie with title, year.\n"

    url = "https://api.openai.com/v1/chat/completions"

    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer " + API_KEY
    }
    params = {
        "model": model_type,
        "messages": [{"role": "system", "content": sys_msg}, 
                        {"role": "user", "content": prompt}
                        ],
        "max_tokens": 1024,
        "temperature": 0.7,
        "stream": False
    }
    for retry in range(5):
        try:
            response = requests.post(url=url, headers=headers, json=params)
            message = response.json()

            if response.status_code != 200 or 'choices' not in message:
                raise ValueError(f"Invalid response: {message}")

            content = message['choices'][0]['message']['content']
            print(f"content: {content.strip()}")

            samples = content.strip().split("::")
            if len(samples) != 2:
                raise ValueError("Invalid format in response")

            pos_sample = int(samples[0].strip())
            neg_sample = int(samples[1].strip())

            augmented_sample_dict[index] = {0: pos_sample, 1: neg_sample}

            with open(os.path.join(file_path, 'augmented_sample_dict'), 'wb') as f:
                pickle.dump(augmented_sample_dict, f)
            return 0

        except Exception as e:
            print(f"❌ Error at index {index}, retry {retry+1}/5: {e}")
            time.sleep(5)

    print(f"❌ Failed to process index {index} after retries")
    failed_indices.append(index)
    return 1

def main(dataset="books"):# netflix, movielens
    if dataset == "netflix":
        file_path = "./LLMRec/LLMRec_c/" + dataset + "/netflix_valid_item/"
    elif dataset == "movielens":
        file_path = "./LLMRec/LLMRec_c/" + dataset + "/ml100k_llmrec_format/"
    elif dataset == "books":
        file_path = "./data/books/books_llmrec_format/"
        
    model_type = "gpt-4o"

    candidate_indices = pickle.load(open(os.path.join(file_path, 'candidate_indices'), 'rb'))
    candidate_indices_dict = {i: candidate_indices[i] for i in range(candidate_indices.shape[0])}

    ### read adjacency_list
    train_mat = pickle.load(open(os.path.join(file_path, 'train_mat'),'rb'))
    adjacency_list_dict = {}
    for index in range(train_mat.shape[0]):
        _, data_y = train_mat[index].nonzero()
        adjacency_list_dict[index] = data_y
    
    ### read item_attribute
    if dataset == "netflix":
        toy_item_attribute = pd.read_csv(os.path.join(file_path, 'item_attribute.csv'), names=['id', 'year', 'title'])
    elif dataset == "movielens":
        toy_item_attribute = pd.read_csv(os.path.join(file_path, 'item_attribute.csv'), names=['id', 'year', 'title', 'genre'])
    elif dataset == "books":
        toy_item_attribute = pd.read_csv(os.path.join(file_path, 'item_attribute.csv'), names=['id', 'brand', 'title', 'category'])
    
    ### write augmented dict
    augmented_sample_dict = {}
    aug_path = os.path.join(file_path, "augmented_sample_dict")
    with open(aug_path, 'wb') as f:
        pickle.dump(augmented_sample_dict, f)

    failed_indices = []
    for index in range(len(adjacency_list_dict)):
        LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, model_type, augmented_sample_dict, file_path, failed_indices, dataset)

    # Save failed indices
    if failed_indices:
        with open(os.path.join(file_path, 'failed_ui_aug_indices.pkl'), 'wb') as f:
            pickle.dump(failed_indices, f)
        print(f"❗ {len(failed_indices)} indices failed and saved to failed_ui_aug_indices.pkl")
    else:
        print("✅ All indices processed successfully")
        
if __name__ == '__main__':
    main()