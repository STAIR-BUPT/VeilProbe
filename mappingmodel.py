import os
import random
import gc
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    BertTokenizer,
    BertConfig,
    EncoderDecoderModel,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
import wandb

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Ensure reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Read CSV data and convert to (prefix, generated text, label) tuples
def read_data(file_path):
    df = pd.read_csv(file_path)
    pairs = []
    for _, row in df.iterrows():
        prefix = str(row['Prefix'])
        label = row['Label']
        for col in ['Generate1', 'Generate2', 'Generate3']:
            pairs.append((prefix, str(row[col]), label))
    return pairs

# Callback to log loss during training using wandb
class LogLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            wandb.log({"train_loss": logs['loss'], "step": state.global_step})
    def on_epoch_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()

# Custom Dataset for Seq2Seq task
class Seq2SeqDataset(Dataset):
    def __init__(self, input_ids, target_ids, labels):
        self.input_ids = input_ids
        self.target_ids = target_ids
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'labels': self.target_ids[idx],
            'class_label': self.labels[idx]
        }

# Tokenize and encode input/output pairs
def preprocess_data(data, tokenizer, max_length=512):
    inputs = tokenizer([x for x, y, l in data], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    targets = tokenizer([y for x, y, l in data], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    labels = [l for _, _, l in data]
    return inputs.input_ids, torch.tensor(targets.input_ids, dtype=torch.int64), labels

# Training function using HuggingFace Trainer
def train_seq2seq(data, model, tokenizer, output_dir, num_epochs):
    input_ids, target_ids, labels = preprocess_data(data, tokenizer)
    dataset = Seq2SeqDataset(input_ids, target_ids, labels)

    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = tokenizer.vocab_size

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        report_to="wandb",
        no_cuda=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        callbacks=[LogLossCallback()]
    )

    print("Starting training...")
    trainer.train()

    if output_dir:
        print("Saving model and tokenizer...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'epoch': num_epochs
        }, os.path.join(output_dir, "training_state.pth"))

# Main function
def main():
    set_seed(42)

    # Paths to CSV files (can be modified)
    data_files = [
        'generated_data_xxxx.csv',
    ]
    
    # Load and combine all data
    all_data = []
    for path in data_files:
        all_data.extend(read_data(path))

    # Encoder and decoder configuration
    encoder_config = BertConfig(
        vocab_size=30522,
        hidden_size=512,
        num_attention_heads=8,
        num_hidden_layers=6,
        intermediate_size=2048,
        max_position_embeddings=512,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1
    )

    decoder_config = BertConfig(
        vocab_size=30522,
        hidden_size=512,
        num_attention_heads=8,
        num_hidden_layers=6,
        intermediate_size=2048,
        max_position_embeddings=512,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        is_decoder=True,
        add_cross_attention=True
    )

    # Build encoder-decoder model
    encoder = AutoModel.from_config(encoder_config)
    decoder = AutoModelForCausalLM.from_config(decoder_config)
    model = EncoderDecoderModel(encoder=encoder, decoder=decoder).to(device)

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Output directory for the trained model
    output_dir = '/transformermodel_xxx'

    # Initialize wandb
    wandb.init(project="seq2seq")

    # Train the model
    train_seq2seq(all_data, model, tokenizer, output_dir, num_epochs=2)

    print("Training completed. Model saved.")
    wandb.finish()

if __name__ == "__main__":
    main()