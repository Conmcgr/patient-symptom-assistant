import pandas as pd
import re
import os
import json
from sklearn.model_selection import train_test_split

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
raw_data_path = os.path.join(project_root, "data", "raw", "Symptom2Disease.csv")
cleaned_data_dir = os.path.join(project_root, "data", "cleaned")

os.makedirs(cleaned_data_dir, exist_ok=True)

print("Loading and preprocessing data...")
df = pd.read_csv(raw_data_path)

df.drop('Index', axis=1, inplace=True)
df.columns = ['disease', 'symptoms']

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s.,;:()/+-]', '', str(text))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['symptoms'] = df['symptoms'].apply(clean_text)
df['disease'] = df['disease'].apply(clean_text)

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

print(f"Dataset size after cleaning: {len(df)} entries")
print("\nSample data:")
print(df.head())

print("\nMost common diseases:")
print(df['disease'].value_counts().head(10))

formatted_data = []
for _, row in df.iterrows():
    formatted_data.append({
        "instruction": "Based on the following symptoms, provide a possible medical diagnosis:",
        "input": row['symptoms'],
        "output": row['disease']
    })

train_data, temp_data = train_test_split(formatted_data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print(f"\nSplit sizes: Train={len(train_data)}, Validation={len(val_data)}, Test={len(test_data)}")

train_path = os.path.join(cleaned_data_dir, "train.jsonl")
val_path = os.path.join(cleaned_data_dir, "val.jsonl")
test_path = os.path.join(cleaned_data_dir, "test.jsonl")

for data, path in [(train_data, train_path), (val_data, val_path), (test_data, test_path)]:
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

print(f"\nData saved to {cleaned_data_dir}")
print(f"- Train data: {train_path}")
print(f"- Validation data: {val_path}")
print(f"- Test data: {test_path}")
