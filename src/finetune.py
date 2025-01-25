import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

training_file = os.path.join(PROJECT_ROOT, "data", "cleaned", "train.jsonl")
validation_file = os.path.join(PROJECT_ROOT, "data", "cleaned", "val.jsonl")

data = load_dataset("json", data_files={"train": training_file, "validation": validation_file})
NUM_EPOCHS = 5
LEARNING_RATE = 5e-5
BATCH_SIZE = 8
MODEL_SAVE_PATH = "../models/lora_adapter"
LOG_DIR = "../logs"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.config.pad_token_id = model.config.eos_token_id

def tokenize(batch):
    inputs = tokenizer(
        batch["input"], 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )
    outputs = tokenizer(
        batch["output"], 
        truncation=True, 
        padding="max_length", 
        max_length=32
    )
    
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": outputs["input_ids"]
    }

tokenized_data = data.map(tokenize, batched=True)
training_data = tokenized_data["train"]
validation_data = tokenized_data["validation"]

def collate_fn(batch):
    input_ids = torch.tensor([item['input_ids'] for item in batch])
    attention_mask = torch.tensor([item['attention_mask'] for item in batch])
    labels = torch.tensor([item['labels'] for item in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

training_loader = DataLoader(
    training_data, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    collate_fn=collate_fn
)
validation_loader = DataLoader(
    validation_data, 
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn
)

model = GPT2LMHeadModel.from_pretrained("gpt2")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, lora_config)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

writer = SummaryWriter(log_dir="./logs")

writer = SummaryWriter(log_dir=LOG_DIR)
best_val_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0
    train_pbar = tqdm(training_loader, desc=f'Training Epoch {epoch + 1}')
    
    for batch in train_pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_pbar.set_postfix({'loss': loss.item()})

    avg_train_loss = train_loss / len(training_loader)
    print(f"\nEpoch {epoch + 1}: Average Training Loss = {avg_train_loss:.4f}")
    writer.add_scalar("Loss/Train", avg_train_loss, epoch)

    model.eval()
    val_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        val_pbar = tqdm(validation_loader, desc=f'Validation Epoch {epoch + 1}')
        for batch in val_pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()
            
            # Generate predictions
            pred = model.generate(
                input_ids=input_ids,
                max_length=32,
                num_return_sequences=1,
                no_repeat_ngram_size=2
            )
            
            predictions.extend(pred.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
            val_pbar.set_postfix({'loss': loss.item()})

    avg_val_loss = val_loss / len(validation_loader)
    print(f"Epoch {epoch + 1}: Average Validation Loss = {avg_val_loss:.4f}")
    writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model.save_pretrained(os.path.join(MODEL_SAVE_PATH, "best_model"))
        print(f"Saved best model with validation loss: {best_val_loss:.4f}")

model.save_pretrained(os.path.join(MODEL_SAVE_PATH, "final_model"))
writer.close()