import threading
import openai
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
API_KEY = os.environ.get("OPENAI_API_KEY")  # 없으면 None

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
    elif dataset.lower() == "movielens":
        history_string = "User history:\n"
        for index in item_list[-10:]:
            title = item_attribute['title'][index]
            year = item_attribute['year'][index]
            genre = item_attribute['genre'][index]
            history_string += f"([{index}] {year}, {title}, {genre})\n"

        output_format = (
            "Please output the following infomation of user, output format(json):\n"
            "{'age': '...', 'gender': '...', 'occupation': '...', 'continent': '...', 'human race': '...', 'preference summarization': '...'}\n"
        )
        prompt = (
            "You are User Analyst. Based on the user history below (each entry contains title, year, and genre), \
            predict the following: age, gender, occupation, continent, human race, and one-sentence summary of preferences (under 50 words, like/dislike info). \
            Output only the following JSON (wrap all values with single quotes):, \n\n" + history_string + output_format
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
    return prompt
    
def LLM_request(toy_item_attribute, adjacency_list_dict, index, model_type, augmented_user_profiling_dict, file_path, error_cnt, dataset, temp):
    if index in augmented_user_profiling_dict:
        return 0
    try:
        print(f"🟢 Index: {index}")
        prompt = construct_user_prompt(toy_item_attribute, adjacency_list_dict[index], dataset)

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + API_KEY
        }

        payload = {
            "model": model_type,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temp,
            # "max_tokens": 1000,
            "top_p": 0.1,
            "stream": False
        }

        response = requests.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            print(f"API Error {response.status_code}: {response.text}")
            return 1
        
        message = response.json()
        content = message['choices'][0]['message']['content']
        print(f" Response received: {content[:100]}...")

        augmented_user_profiling_dict[index] = content
        pickle.dump(augmented_user_profiling_dict, open(file_path, 'wb'))
        return 0
    except (requests.exceptions.RequestException, KeyError, IndexError, json.JSONDecodeError) as e:
        print(f" Error occurred: {e}")
    except Exception as ex:
        print(f" Unexpected error: {ex}")
    error_cnt += 1
    if error_cnt >= 5:
        print("Too many errors. Aborting.")
        return 1
    time.sleep(3)
    return LLM_request(toy_item_attribute, adjacency_list_dict, index, model_type, augmented_user_profiling_dict, file_path, error_cnt, dataset, temp)

# embedding
def LLM_embedding_request(augmented_user_profiling_dict, index, model_type, augmented_user_init_embedding, file_path):
    if index in augmented_user_init_embedding:
        return 0
    else:
        try: 
            url = "https://api.openai.com/v1/embeddings"
            headers={
                "Authorization": "Bearer " + API_KEY,
            }
            params={
            "model": model_type,
            "input": augmented_user_profiling_dict[index]
            }
            response = requests.post(url, headers=headers, json=params)
            if response.status_code != 200:
                print(f"HTTP Error {response.status_code}: {response.text}")
                return 1
            
            message = response.json()
            embedding = message['data'][0]['embedding']
            augmented_user_init_embedding[index] = np.array(embedding)
            
            #print("Raw response:", message)            
            # print(content)
            #print(index)

            with open(os.path.join(file_path, 'augmented_user_init_embedding'), 'wb') as f:
                pickle.dump(augmented_user_init_embedding, f)
            print(f"✅ Embedded index {index}")
            return 0
        

        # except ValueError as e:
        except requests.exceptions.RequestException as e:
            print("An HTTP error occurred:", str(e))
            time.sleep(5)
        except ValueError as ve:
            print("An error occurred while parsing the response:", str(ve))
            time.sleep(5)
            LLM_embedding_request(augmented_user_profiling_dict, index, model_type, augmented_user_init_embedding, file_path)
        except KeyError as ke:
            print("An error occurred while accessing the response:", str(ke))
            time.sleep(5)
            LLM_embedding_request(augmented_user_profiling_dict, index, model_type, augmented_user_init_embedding, file_path)
        except Exception as ex:
            print("An unknown error occurred:", str(ex))
            time.sleep(5)
        
        return 1


## step1: generate user profiling ################################################################################## 
def step1(file_path, g_model_type, start_id, error_cnt, dataset, temp):
    augmented_user_profiling_dict = {}  
    dict_path = os.path.join(file_path, f"augmented_user_profiling_dict")
    with open(dict_path, 'wb') as f:
        pickle.dump(augmented_user_profiling_dict, f)
    
    # read item_attribute
    if dataset.lower() == "netflix":
        toy_item_attribute = pd.read_csv(os.path.join(file_path, 'item_attribute.csv'), names=['id', 'year', 'title'])
    elif dataset.lower() == "movielens":
        toy_item_attribute = pd.read_csv(os.path.join(file_path, 'item_attribute.csv'))  # ✅ 헤더 자동 감지
    elif dataset.lower() == "books":
        toy_item_attribute = pd.read_csv(os.path.join(file_path, 'item_attribute.csv'), names=['id', 'brand', 'title', 'category']) 
    else:
        raise ValueError(f"Unknown dataset type: {dataset}")
    
    train_mat = pickle.load(open(os.path.join(file_path, 'train_mat'), 'rb'))
    ### read adjacency_list
    adjacency_list_dict = {}
    for index in range(train_mat.shape[0]):
        data_x, data_y = train_mat[index].nonzero()
        adjacency_list_dict[index] = data_y

    for index in range(start_id, len(adjacency_list_dict.keys())):
        print(index)
        # # make prompting
        LLM_request(toy_item_attribute, adjacency_list_dict, index, g_model_type, augmented_user_profiling_dict, dict_path, error_cnt, dataset, temp)
    # "claude", "chatglm-6b", "hambuger-13b", "baichuan-7B", "gpt-4", "gpt-4"
    ## step1: generate user profiling################################################################################## 

### step2: generate user embedding ################################################################################## 
def step2(emb_model, file_path):
    augmented_user_profiling_dict = pickle.load(open(file_path + 'augmented_user_profiling_dict','rb'))
    ### write augmented_user_init_embedding
    augmented_user_init_embedding = {}  
    pickle.dump(augmented_user_init_embedding, open(file_path + 'augmented_user_init_embedding','wb'))

    for user_id in augmented_user_profiling_dict.keys():
        LLM_embedding_request(augmented_user_profiling_dict, user_id, emb_model, augmented_user_init_embedding, file_path)

# # ### step3: get user embedding ################################################################################## 
def step3(file_path):
    embed_dict = pickle.load(open(os.path.join(file_path, 'augmented_user_init_embedding'), 'rb'))
    train_mat = pickle.load(open(file_path + 'train_mat', 'rb'))
    n_users = train_mat.shape[0]

    final_matrix = []
    for i in range(n_users):
        if i in embed_dict:
            final_matrix.append(embed_dict[i])
        else:
            print(f"⚠️ Missing user {i} – inserting zero vector")
            final_matrix.append(np.zeros(1536))
    
    final_array = np.array(final_matrix)
    with open(file_path + 'augmented_user_init_embedding_final', 'wb') as f:
        pickle.dump(final_array, f)

    print(f"✅ Final user embedding saved: shape = {final_array.shape}")


def main(dataset):
    openai.api_base = "https://api.openai.com/v1"
    if dataset == "netflix":
        file_path = "./LLMRec/LLMRec_c/" + dataset + "/netflix_valid_item/"
    elif dataset == "movielens":
        file_path = "./LLMRec/LLMRec_c/" + dataset + "/ml100k_llmrec_format/"
    elif dataset == "books":
        file_path = "./data/books/books_llmrec_format/"
    
    max_threads = 5
    cnt = 0
    start_id = 0
    gen_model_type_4o = "gpt-4o" # gpt-3.5-turbo, gpt-4o
    embedding_model = "text-embedding-ada-002"
    ## "claude", "chatglm-6b", "hambuger-13b", "baichuan-7B", "gpt-4", "gpt-4o"
    error_cnt = 0
    
    temperatures = [0, 0.1, 0.5, 1.2]
    
    # for temp in temperatures:
    #     step1(file_path, gen_model_type_4, start_id, error_cnt, dataset, temp)
    #     step1(file_path, gen_model_type_turbo, start_id, error_cnt, dataset, temp)
    
    step1(file_path, gen_model_type_4o, start_id, error_cnt, dataset, temp = 0.6)
    step2(embedding_model, file_path)
    step3(file_path)

if __name__ == '__main__':
    main()