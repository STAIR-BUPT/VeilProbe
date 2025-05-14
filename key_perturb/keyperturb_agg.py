import torch
import random
import pandas as pd
from tqdm import tqdm
import re
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM, GPT2Tokenizer, GPT2Model,GPT2LMHeadModel
from utils import load_model, get_template
from interpreter import Interpreter


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model_names = ["gpt2","gpt2-medium","gpt2-large"]
models = []
tokenizers = []
interpreters = []

for model_name in model_names:
    model, tokenizer, block_name, embedding_name, embed_token_name, _, _= load_model(model_name)
    interpreter = Interpreter(model, block_name, embed_token_name, embed_token_name)
    models.append(model)
    tokenizers.append(tokenizer)
    interpreters.append(interpreter)


modified_model_name = 'bert-base-cased'
local_modified_model_path = './models/bert-base-cased'


tokenizer_modified = AutoTokenizer.from_pretrained(local_modified_model_path)
# tokenizer_modified.save_pretrained(local_modified_model_path)

model_modified = AutoModelForMaskedLM.from_pretrained(local_modified_model_path)
# model_modified.save_pretrained(local_modified_model_path)

unmasker = pipeline("fill-mask", model=model_modified, tokenizer=tokenizer_modified)

def perturb_sentence_with_mlm(sentence, max_indices, tokens, tokenizer):
    sentence = str(sentence)
    """
    Perturb the sentence by replacing selected tokens with predictions from a Masked Language Model (MLM).

    Args:
        sentence (str): Original input sentence.
        max_indices (list): Indices of tokens to be replaced.
        tokens (list): Tokenized input as a list of strings.
        tokenizer: Hugging Face tokenizer.

    Returns:
        str: The perturbed sentence.
    """
    perturbed_tokens = tokens.copy()

    # Define a regex pattern to detect special symbols
    special_symbol_pattern = re.compile(r"[^a-zA-Z0-9\-]")

    for idx in max_indices:
        word = tokenizer.convert_tokens_to_string([tokens[idx]]).strip()

        # Replace the target word with [MASK] in the sentence
        masked_sentence = sentence.replace(word, "[MASK]", 1)
        # Ensure `[MASK]` is in the sentence
        if "[MASK]" not in masked_sentence:
            print(f"Skipping token '{word}' as it could not be replaced with [MASK]")
            continue

        # Get predictions from the MLM
        predictions = unmasker(masked_sentence)  # Use the preloaded `unmasker`

        # Filter top predictions to exclude the original word and special symbols
        top_predictions = []
        for pred in predictions[:5]:
            try:
                token_str = pred["token_str"].strip()
                if token_str != word and not special_symbol_pattern.search(token_str):
                    top_predictions.append(token_str)
            except (TypeError, KeyError) as e:
                print(f"Skipping due to error: {e}, pred: {pred}")
                continue
        
        if top_predictions:
            replacement = random.choice(top_predictions)

            # Check if the token needs a preceding space
            if tokens[idx].startswith("Ġ"):
                replacement = "Ġ" + replacement  # Add a space to the replacement if needed
            perturbed_tokens[idx] = replacement

    # Join the perturbed tokens to form the final sentence
    perturbed_sentence = tokenizer.convert_tokens_to_string(perturbed_tokens).strip()
    return perturbed_sentence

def compute_attributions_for_sentence(sentence, models, tokenizers, interpreters, top_k=26):
    token_scores = {}
    sentence = str(sentence)
    for model, tokenizer, interpreter in zip(models, tokenizers, interpreters):
        template = get_template(model_names[models.index(model)])
        input_text = f"{template['prefix']}{sentence.strip()}{template['postfix']}"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        beams = 10
        max_new_tokens = 10
        attributions, _ = interpreter.interpret_ours(inputs.input_ids, beams, max_new_tokens, "optimal_transport")
        attribution_scores = torch.stack([d['optimal_transport'] for d in attributions])
        tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        while top_k > attribution_scores.size(0):
            top_k = top_k // 2
        max_scores, max_indices = torch.topk(attribution_scores, k=top_k, dim=0)
        for i, idx in enumerate(max_indices):
            token = tokens[idx.item()]
            if token not in token_scores:
                token_scores[token] = []
            token_scores[token].append((max_scores[i].item(), idx.item()))

    average_scores = {token: sum(score for score, _ in scores) / len(scores) for token, scores in token_scores.items()}
    choose_token_num = 13
    final_keywords = [
    (
        token, 
        average_scores[token], 
        max(set([idx for _, idx in token_scores[token]]), key=[idx for _, idx in token_scores[token]].count)
    )
    for token in sorted(average_scores, key=average_scores.get, reverse=True)[:choose_token_num]]

    return {"keywords": final_keywords, "average_scores": average_scores}

def process_sentence(sentence, models, tokenizers, interpreters):
    sentence = str(sentence)
    result = compute_attributions_for_sentence(sentence, models, tokenizers, interpreters)

    template = get_template(model_names[0])
    input_text = f"{template['prefix']}{sentence.strip()}{template['postfix']}"

    tokenizer = tokenizers[0]
    inputs = tokenizer(input_text, return_tensors="pt")
    tokens_list = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    max_indices =[idx_list for _, _, idx_list in result['keywords']]

    perturbed_sentence = perturb_sentence_with_mlm(input_text, max_indices, tokens_list, tokenizer)

    return {
        "perturbed_sentence": perturbed_sentence,
        "keywords": result['keywords']
    }

def process_excel_file(input_file, output_file, models, tokenizers, interpreters):
    df = pd.read_excel(input_file)

    if 'Item' not in df.columns:
        raise ValueError("Input Excel file must contain an 'Item' column.")

    df['Perturbed Sentence'] = ""
    df['Key Tokens and Scores'] = ""

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Rows"):
        sentence = row['Item']
        # Skip processing if sentence is None
        if sentence is None or pd.isna(sentence):
            continue

        sentence = str(sentence)
        result = process_sentence(sentence, models, tokenizers, interpreters)

        df.at[index, 'Perturbed Sentence'] = result['perturbed_sentence']
        key_tokens_str = ", ".join([f"{token}: {score:.4f}" for token, score, _ in result['keywords']])
        df.at[index, 'Key Tokens and Scores'] = key_tokens_str

    df.to_excel(output_file, index=False)

if __name__ == "__main__":
    input_excel = "/path/to/file/booktection128_nonmember.xlsx"
    output_excel = "/path/to/file/booktection128_keyperturb_non_member.xlsx"

    process_excel_file(input_excel, output_excel, models, tokenizers, interpreters)
