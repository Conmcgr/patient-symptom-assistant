import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from tqdm import tqdm
import os

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
training_file = os.path.join(PROJECT_ROOT, "data", "cleaned", "train.jsonl")
validation_file = os.path.join(PROJECT_ROOT, "data", "cleaned", "val.jsonl")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "lora_adapter")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# Create directories
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def save_model(model, tokenizer, lora_config, path):
    os.makedirs(path, exist_ok=True)
    
    # Save the model state
    model.save_pretrained(path)
    
    # Save the adapter config explicitly
    config_path = os.path.join(path, "adapter_config.json")
    lora_config.save_pretrained(path)
    
    # Save tokenizer
    tokenizer.save_pretrained(path)

# Load dataset
data = load_dataset("json", data_files={"train": training_file, "validation": validation_file})

# Initialize tokenizer with correct settings
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'  # Set padding to left side for generation

# Initialize model
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.config.pad_token_id = model.config.eos_token_id
model.resize_token_embeddings(len(tokenizer))

def tokenize(batch):
    # Tokenize inputs
    inputs = tokenizer(
        batch["input"],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_attention_mask=True
    )
    
    # Tokenize outputs and shift them for GPT2 training
    with tokenizer.as_target_tokenizer():
        outputs = tokenizer(
            batch["output"],
            truncation=True,
            padding="max_length",
            max_length=128  # Make this match input length
        )
    
    # Create labels with -100 for input tokens
    labels = [-100] * len(inputs["input_ids"])  # Initialize with -100
    labels[-len(outputs["input_ids"]):] = outputs["input_ids"]  # Add output tokens at the end
    
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels
    }

def collate_fn(batch):
    input_ids = torch.tensor([item['input_ids'] for item in batch])
    attention_mask = torch.tensor([item['attention_mask'] for item in batch])
    labels = torch.tensor([item['labels'] for item in batch])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# Update LoRA config
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj"],
    fan_in_fan_out=True  # Add this to address the warning
)



# Process data
tokenized_data = data.map(tokenize, batched=True)
training_data = tokenized_data["train"]
validation_data = tokenized_data["validation"]

# Setup data loaders
BATCH_SIZE = 8
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

def check_data_sample(data_loader, tokenizer):
    """Print a few examples to verify data formatting"""
    batch = next(iter(data_loader))
    for i in range(min(3, len(batch['input_ids']))):
        print("\nExample", i+1)
        print("Input:", tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True))
        # Find where labels start (first non -100 value)
        label_start = [j for j, x in enumerate(batch['labels'][i]) if x != -100][0]
        print("Expected Output:", tokenizer.decode(batch['labels'][i][label_start:], skip_special_tokens=True))
        print("-" * 50)

print("\nChecking training data samples:")
check_data_sample(training_loader, tokenizer)

# Initialize model with LoRA
model = get_peft_model(model, lora_config)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
writer = SummaryWriter(log_dir=LOG_DIR)
best_loss = float('inf')
NUM_EPOCHS = 5

if device.type == "cpu":
    print("Training on CPU. Reducing batch size and learning rate for stability.")
    global BATCH_SIZE, optimizer
    BATCH_SIZE = 2
    training_loader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for epoch in range(NUM_EPOCHS):
    # Training phase
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
        if torch.isnan(loss) or torch.isinf(loss):
            print("WARNING: Skipping batch due to nan or inf loss.")
            continue
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_pbar.set_postfix({'loss': loss.item()})

    avg_train_loss = train_loss / len(training_loader)
    print(f"\nEpoch {epoch + 1}: Average Training Loss = {avg_train_loss:.4f}")
    writer.add_scalar("Loss/Train", avg_train_loss, epoch)
    
    # Save model if loss improved
    if avg_train_loss < best_loss:
        best_loss = avg_train_loss
        save_model(model, tokenizer, lora_config, os.path.join(MODEL_SAVE_PATH, "best_model"))
        print(f"Saved model with loss: {best_loss:.4f}")

# Save final model
save_model(model, tokenizer, lora_config, os.path.join(MODEL_SAVE_PATH, "final_model"))
writer.close()




