import os
import gc
import random
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import ttest_ind

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split

from transformers import (
    EncoderDecoderModel, 
    BertTokenizer,
    DataCollatorForSeq2Seq
)

import wandb

from protonetIB import PrototypicalNet

# Set device and seed
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# Read data
def read_data(file_path, domain, label):
    df = pd.read_excel(file_path)
    col = 'Item' if domain == 'original' else 'Perturbed Sentence'
    return [(str(row[col]), label) for _, row in df.iterrows()]

# Feature extraction and processing
def extract_and_combine_features(model, data, tokenizer, device):
    model.eval()
    combined_features_dict = {}
    max_length = 512

    with torch.no_grad():
        combined_features_total = None

        for i, (input_text, label) in enumerate(tqdm(data, desc="Extracting features")):
            inputs = tokenizer(input_text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt").to(device)
            decoder_token = tokenizer.cls_token or tokenizer.bos_token
            decoder_inputs = tokenizer(decoder_token, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(device)

            encoder_out = model.encoder(**inputs, output_hidden_states=True)
            decoder_out = model.decoder(input_ids=decoder_inputs['input_ids'], encoder_hidden_states=encoder_out.last_hidden_state, output_hidden_states=True)

            encoder_hidden = torch.cat(encoder_out.hidden_states, dim=-1).cpu().numpy()
            decoder_hidden = torch.cat(decoder_out.hidden_states, dim=-1).cpu().numpy()

            combined_features = np.concatenate((encoder_hidden, decoder_hidden), axis=-1)

            if i == 0:
                combined_features_total = np.empty((len(data), *combined_features.shape), dtype=np.float32)
            combined_features_total[i] = combined_features
            combined_features_dict[i] = {'text_out': input_text, 'label': label}

    return combined_features_dict, combined_features_total

# Feature pruning
def feature_significance_ttest(features, labels, p_threshold):
    valid_features = features[labels != -1]
    valid_labels = labels[labels != -1]
    group_0 = valid_features[valid_labels == 0]
    group_1 = valid_features[valid_labels == 1]
    significant = [i for i in range(valid_features.shape[1]) if ttest_ind(group_0[:, i], group_1[:, i], equal_var=False)[1] < p_threshold]
    return significant

def prune_features(train, test, labels, p_threshold=0.01):
    sig_indices = feature_significance_ttest(train, labels, p_threshold)
    mask = np.zeros(train.shape[1], dtype=bool)
    mask[sig_indices] = True
    train[:, ~mask] = 0
    test[:, ~mask] = 0
    return train, test, sig_indices


def combine_and_process_features(features_original, diff_abs):
    combined_features = torch.cat((features_original, diff_abs), dim=-1)
    return combined_features

# Main entry

def main():
    print("Loading and processing data...")

    # Load original mapping model
    model1_path = 'transformermodel_booktection128_keyperturb_gpt3.5-turbo-instruct_e2_batch2'
    tokenizer = BertTokenizer.from_pretrained(model1_path)
    model = EncoderDecoderModel.from_pretrained(model1_path).to(device)

    # Read data
    file_path_member = 'data/booktection_seq128.xlsx'
    file_path_non_member = 'data/booktection_non_seq128.xlsx'
    data = read_data(file_path_member, 'original', 1) + read_data(file_path_non_member, 'original', 0)

    # Extract features from original model
    features_dict, features_original = extract_and_combine_features(model, data, tokenizer, device)
    labels = np.array([f['label'] for f in features_dict.values()])
    texts_all = [f['text_out'] for f in features_dict.values()]
    features_original = np.mean(features_original, axis=2).squeeze(1)

    # Load perturbed mapping model and perturbed data
    model_perturb_path = 'transformermodel_booktection128_keyperturb_gpt3.5-turbo-instruct_e2_batch2'
    tokenizer_perturb = BertTokenizer.from_pretrained(model_perturb_path)
    model_perturb = EncoderDecoderModel.from_pretrained(model_perturb_path).to(device)

    file_path_perturb_member = 'perturbed data/booktection128_keyperturb_member.xlsx'
    file_path_perturb_non_member = 'perturbed data/booktection128_keyperturb_non_member.xlsx'
    data_perturb = read_data(file_path_perturb_member, 'perturb', 1) + read_data(file_path_perturb_non_member, 'perturb', 0)

    features_dict_perturb, features_perturb = extract_and_combine_features(model_perturb, data_perturb, tokenizer_perturb, device)
    features_perturb = np.mean(features_perturb, axis=2).squeeze(1)

    # Train/test split
    test_size = 0.99
    ftr_o_train, ftr_o_test, ftr_p_train, ftr_p_test, y_train, y_test, text_train, text_test = train_test_split(
        features_original, features_perturb, labels, texts_all, test_size=test_size, stratify=labels)
    
    
    print("Ground Truth number is:", int(len(data)*(1-test_size)))
    # Compute and prune absolute differences
  
    diff_train = np.abs(ftr_p_train - ftr_o_train)
    diff_test = np.abs(ftr_p_test - ftr_o_test)
    diff_train, diff_test, sig_indices = prune_features(diff_train, diff_test, y_train)

    # Combine features
    ftr_o_train_tensor = torch.tensor(ftr_o_train)
    diff_train_tensor = torch.tensor(diff_train)
    features_train = combine_and_process_features(ftr_o_train_tensor, diff_train_tensor).cpu().numpy()

    ftr_o_test_tensor = torch.tensor(ftr_o_test)
    diff_test_tensor = torch.tensor(diff_test)
    features_test = combine_and_process_features(ftr_o_test_tensor, diff_test_tensor).cpu().numpy()

    # Initialize and train prototype network
    proto_net = PrototypicalNet(
        X_train=features_train,
        X_test=features_test,
        y_train=y_train,
        y_test=y_test,
        input_dim=features_train.shape[1],
        hidden_dim=64,
        bottleneck_dim=16,
        input_texts_test=text_test,
        device=device
    )

    wandb.init(project="Prototypical_Net", config={"epochs": 50, "epoch_size": 10})
    proto_net.train(50, 10, features_train, features_test, y_train, y_test, text_test)
    wandb.finish()

if __name__ == "__main__":
    main()
