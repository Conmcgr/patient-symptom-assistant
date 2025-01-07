import pandas as pd
import re

df = pd.read_csv("../data/raw/Symptom2Disease.csv")

#input --> natural language patient reported symptoms, output --> diagnosis

df.drop('Index', axis=1, inplace=True)
df.columns = ['output', 'input']
df['input'] = df['input'].str.replace(r'[^a-zA-Z0-9\s.,]', '', regex=True)
df['input'] = df['input'].str.lower()
df['output'] = df['output'].str.replace(r'[^a-zA-Z0-9\s.,]', '', regex=True)
df['output'] = df['output'].str.lower()

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

print(df.head())
print(df['output'].value_counts())

train_df = df.sample(frac=0.9, random_state=42)  # 90% train
val_df = df.drop(train_df.index)  # 10% validation

train_df.to_json('../data/cleaned/train.jsonl', orient='records', lines=True)
val_df.to_json('../data/cleaned/val.jsonl', orient='records', lines=True)

