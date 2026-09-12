import torch
from transformers import AutoTokenizer
from datasets import load_dataset , Dataset
import pandas as pd
from transformers import DataCollatorWithPadding
data = {
    'text': [
        "I love machine learning",
        "Deep learning is amazing",
        "This is terrible",
        "I hate bugs in my code",
        "HuggingFace is fantastic",
        "Financial problems are stressful",
        "I will succeed in freelancing",
        "This project is very bad"
    ],
    'label': [1, 1, 0, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)
df
dataset = Dataset.from_pandas(df)
dataset
model = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def tokenize_fun(examples):
  return tokenizer(examples['text'], padding='max_length', truncation=True)
data_map=dataset.map(tokenize_fun , batched= True )
data_map.column_names
data_map = data_map.remove_columns(['text'])
data_map.column_names
removed=data_map.rename_column('label' , 'labels')
removed.column_names
removed.set_format("torch")
type(removed)
