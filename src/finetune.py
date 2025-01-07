import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

training_file = "../data/cleaned/train.jsonl"
validation_file = "../data/cleaned/val.jsonl"
data = load_dataset("json", data_files={"train": training_file, "validation": validation_file})

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def tokenize(batch):
    inputs = tokenizer(batch["input"], truncation=True, padding="max_length", max_length=128)
    outputs = tokenizer(batch["output"], truncation=True, padding="max_length", max_length=32)
    return {"input_ids": torch.tensor(inputs["input_ids"]), "labels": torch.tensor(outputs["input_ids"])}

tokenized_data = data.map(tokenize, batched=True)
training_data = tokenized_data["train"]
validation_data = tokenized_data["validation"]

training_loader = DataLoader(training_data, batch_size=8, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=8)
