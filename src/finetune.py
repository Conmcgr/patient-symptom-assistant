import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)
import evaluate
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Fine-tune a medical diagnosis model")
parser.add_argument("--base_model", type=str, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", 
                    help="Base model to use for fine-tuning")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
parser.add_argument("--lora_r", type=int, default=16, help="LoRA attention dimension")
parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate")
parser.add_argument("--eval_steps", type=int, default=100, help="Evaluation steps")
parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint steps")
parser.add_argument("--warmup_steps", type=int, default=100, help="Learning rate warmup steps")
parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
parser.add_argument("--use_8bit", action="store_true", help="Use 8-bit quantization for training")
parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization for training")
parser.add_argument("--seed", type=int, default=42, help="Random seed")

args = parser.parse_args([])

torch.manual_seed(args.seed)
np.random.seed(args.seed)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
data_dir = os.path.join(project_root, "data", "cleaned")
model_dir = os.path.join(project_root, "models")
output_dir = os.path.join(model_dir, "medical_diagnosis_model")

os.makedirs(output_dir, exist_ok=True)

print("Loading datasets...")
data_files = {
    "train": os.path.join(data_dir, "train.jsonl"),
    "validation": os.path.join(data_dir, "val.jsonl"),
    "test": os.path.join(data_dir, "test.jsonl")
}
dataset = load_dataset("json", data_files=data_files)

print(f"Loading base model: {args.base_model}")
model_kwargs = {}
if args.use_8bit:
    model_kwargs["load_in_8bit"] = True
elif args.use_4bit:
    model_kwargs["load_in_4bit"] = True
    model_kwargs["bnb_4bit_compute_dtype"] = torch.float16

tokenizer = AutoTokenizer.from_pretrained(args.base_model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    **model_kwargs
)

if args.use_8bit or args.use_4bit:
    model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    target_modules=["query", "key", "value", "dense"],
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

def preprocess_function(examples):
    prompts = [
        f"{instruction}\n\nSymptoms: {input_text}\n\nDiagnosis:" 
        for instruction, input_text in zip(examples["instruction"], examples["input"])
    ]
    
    targets = [f" {output}" for output in examples["output"]]
    
    model_inputs = tokenizer(
        prompts,
        max_length=args.max_length,
        padding="max_length",
        truncation=True,
    )
    
    labels = tokenizer(
        targets,
        max_length=args.max_length,
        padding="max_length",
        truncation=True,
    )
    
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] 
        for label in labels["input_ids"]
    ]
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Processing datasets...")
tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding="max_length",
    max_length=args.max_length,
)

rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    rouge_output = rouge.compute(
        predictions=decoded_preds, 
        references=decoded_labels, 
        use_stemmer=True
    )
    
    bleu_output = bleu.compute(
        predictions=decoded_preds,
        references=[[label] for label in decoded_labels]
    )
    
    results = {
        "bleu": bleu_output["bleu"],
        "rouge1": rouge_output["rouge1"],
        "rouge2": rouge_output["rouge2"],
        "rougeL": rouge_output["rougeL"],
    }
    
    return results

training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="steps",
    eval_steps=args.eval_steps,
    save_strategy="steps",
    save_steps=args.save_steps,
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    weight_decay=0.01,
    warmup_steps=args.warmup_steps,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    greater_is_better=True,
    push_to_hub=False,
    fp16=torch.cuda.is_available(),
    report_to="tensorboard",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

print("Starting training...")
trainer.train()

print("Evaluating on test set...")
test_results = trainer.evaluate(tokenized_datasets["test"])
print(f"Test results: {test_results}")

print("Saving final model...")
trainer.save_model(os.path.join(output_dir, "final_model"))
tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))

print("Saving best model...")
model.save_pretrained(os.path.join(output_dir, "best_model"))
tokenizer.save_pretrained(os.path.join(output_dir, "best_model"))

print("Training complete!")
