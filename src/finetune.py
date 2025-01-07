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

for epoch in range(3):
    model.train()
    train_loss = 0
    for batch in training_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_train_loss = train_loss / len(training_loader)
    print(f"Epoch {epoch + 1}: Average Training Loss = {avg_train_loss:.4f}")

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in validation_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()

    avg_val_loss = val_loss / len(validation_loader)
    print(f"Epoch {epoch + 1}: Average Validation Loss = {avg_val_loss:.4f}")

model.save_pretrained("../models/lora_adapter")