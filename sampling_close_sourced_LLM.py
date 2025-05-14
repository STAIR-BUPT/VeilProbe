import os
import openai
from openai import OpenAI
import requests
import time
import json
import time
import pandas as pd
from tqdm import tqdm
from openpyxl.utils.exceptions import IllegalCharacterError
    
    
API_SECRET_KEY = "sk-"
BASE_URL = "xxxx"

def load_data_from_excel(file_path, column_name):
    df = pd.read_excel(file_path)
    return df[column_name].tolist() 

def get_request(prefix, max_length):

    client = openai.OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)

    while True:  
        try:
            resp = client.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prefix,
                max_tokens=max_length,
                temperature=0.2
            )
            return resp.choices[0].text
        except (openai.APITimeoutError, requests.exceptions.Timeout, TimeoutError) as e:
            time.sleep(5)  
        except TypeError as e:
            print(f"Caught a TypeError: {e}")
            return None  

def save_partial_results(blackbox_info, output_file):
    try:
        df = pd.DataFrame.from_dict(blackbox_info, orient="index")
        df.to_csv(output_file, index=False)
        print(f"Partial results saved to {output_file}")
    except IllegalCharacterError as e:
        print(f"illegal charactor: {e}")



# def get_request(prefix, max_length):

#     client = openai.OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
#     while True:
#         try:
#             resp = client.chat.completions.create(
#                 model="claude-2.1",
#                 max_tokens=max_length,
#                 messages=[
#                 {"role": "system", "content": "You are an assistant that completes the subsequent text based on the prefix."},
#                 {"role": "user", "content": prefix}
#                 ]
#             )
#             return resp.choices[0].message.content
#         except (openai.APITimeoutError, requests.exceptions.Timeout, TimeoutError) as e:
#             time.sleep(5)  
#         except TypeError as e:
#             print(f"Caught a TypeError: {e}")
#             return None 
    
    
  
       
           
def text_out(file1, file2, output_file, max_length):
    print("Constructing features...")
    column_name = "Item"
    items_label_1 = load_data_from_excel(file1, column_name) 
    items_label_0 = load_data_from_excel(file2, column_name) 
    items_sum = items_label_1 + items_label_0
    labels = [1] * len(items_label_1) + [0] * len(items_label_0)
    blackbox_info = {}
    offset = len(blackbox_info)

    items_sum = [str(item) for item in items_sum if item is not None]
    
    for i, item in tqdm(enumerate(items_sum), desc="Processing items"):
        generate_text_list = []
        num_generations = 1
        
        for _ in range(num_generations):
            try:
                generate_text_list_i = get_request(item, max_length)
                if generate_text_list_i is not None:
                    generate_text_list.append(generate_text_list_i)
                else:
                    print(f"None：{item}")
            except Exception as e:
                print(f"None：{item}, error：{e}")
        
        print(f"Generated {len(generate_text_list)} texts for item")

        generate_text_list_remove_prefix = [
            text.removeprefix(item) if text is not None else "" for text in generate_text_list
        ]

        # Create columns for each generated text
        generate_columns = {
            f"Generate{i+1}": generate_text_list_remove_prefix[i] if i < len(generate_text_list_remove_prefix) else ""
            for i in range(num_generations)
        }
        generate_whole_columns = {
            f"Genwhole{i+1}": generate_text_list[i] if i < len(generate_text_list) else ""
            for i in range(num_generations)
        }

        # Store each item’s information in blackbox_info
        blackbox_info[i + offset] = {
            "Item": item,
            "Prefix": item,
            **generate_columns,  # Add generated texts to the dict
            **generate_whole_columns,
            "Label": labels[i]
        }
        if (i + 1) % 500 == 0:  
            save_partial_results(blackbox_info, output_file)

    save_partial_results(blackbox_info, output_file)
    print(f"Data saved to {output_file}")
    return blackbox_info





if __name__ == "__main__":
    file_path_1 = 'data/xxx.xlsx'
    file_path_2 = 'data/xxx_non.xlsx'

    output_file = 'blackbox_xxx_generate_text.csv' 
    
    black_info = text_out(file_path_1, file_path_2, output_file, max_length=512)
    print(black_info)
    
