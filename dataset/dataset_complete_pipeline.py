import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset
from datasets import load_dataset
import torch
from transformers import DataCollatorWithPadding
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer
print(tokenizer.vocab_size)
print(tokenizer.model_max_length)
dataset = load_dataset("imdb")
dataset
print(f"Training rows : {dataset['train'].num_rows}")
print(f"Testing rows : {dataset['test'].num_rows}")
print(dataset.column_names)
print(dataset["train"][0]['text'])  # for just text
print(dataset['train'][0])
positive_label_count = dataset['train'].filter(lambda x: x['label']==1)
print(len(positive_label_count))
negative_label = dataset['test'].filter(lambda x: x['label']==0)
print(len(negative_label))
short_reviews = dataset['train'].filter(lambda x: len(x['text'].split()) < 50)
print(len(short_reviews))
long_reviews = dataset['train'].filter(lambda x: len(x['text'].split()) > 50)
print(len(long_reviews))
# samples
print(dataset['train'][0:10])
print(dataset['train'][30]['label'])
print(dataset['train'][30]['text'])
#
# take one text and tokenize it
single_sample = dataset['train'][0]['text']
single_sample_encoded = tokenizer(single_sample, max_length=512 , padding='max_length' , truncation= True , return_tensors='pt')
print(single_sample_encoded)
print(single_sample_encoded.attention_mask)
print(single_sample_encoded.token_to_chars)

# concepts of padding
text = [
    "i love to code",
    "i love to train a trasformer model from the scratch"
]
for i in text:
  encoded_text = tokenizer(i , add_special_tokens=True , padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors='pt')


  print(f"Decoded text: {tokenizer.decode(encoded_text['input_ids'][0])}")
# concept of the truncation
text = [
    "i love to code the model",
    "i love to train a trasformer model from the scratch"
]
for i in text:
  encoded_text = tokenizer(i, add_special_tokens=True, padding="max_length", max_length=5, truncation='only_first' , return_tensors='pt')
  print(f"Original text: '{i}'")
  print(f"Decoded truncated text: {tokenizer.decode(encoded_text['input_ids'][0])}")
# when there are batches the the problem occue with the padding then we use the datacollectoe with padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# column before the tokenization
print(dataset.column_names)
# tokineze full dataset
def tokenize_dataset(example):
  return tokenizer(example["text"], truncation=True, padding="max_length", return_tensors='pt', max_length=512)

data_map = dataset.map(tokenize_dataset, batched=True)
print(data_map.column_names)
rmv_text = data_map.remove_columns(['text'])
print(rmv_text.column_names)
renme = data_map.rename_column('label', 'labels')
print(renme)
data_map.set_format(type='torch')
print(type(data_map['train'][0]['input_ids']))
print(type(data_map))
print(data_map.column_names)
print(data_map.shape)
print(data_map['train']['input_ids'])
print(data_map['train']['attention_mask'])
# shuffle the dataset
shuffle = data_map.shuffle(seed=42)
small_dataset = data_map['train'].select(range(1000))
print(len(small_dataset))
print(small_dataset)
data = {
    'text': [
        "I finally understood how transformers work after building one from scratch",
        "Financial problems are making my life very difficult right now",
        "My GitHub is growing every single day with new projects",
        "University assignments and freelancing together is very overwhelming",
        "I will get my first Fiverr order very soon",
        "I built a plant disease detector with 99 percent accuracy",
        "I am struggling with money but I never give up",
        "HuggingFace is making my deep learning journey very easy",
        "Sometimes I feel like I cannot handle university and coding together",
        "I am a software engineering student from Lahore Pakistan",
        "Every day I am getting better at machine learning",
        "I do not have enough money to buy proper resources",
        "I completed all six NLP pipelines in just one week",
        "The pressure of financial problems is very stressful",
        "I will earn my first dollar from freelancing very soon"
    ],
    'label': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)
df
custom_dataset = Dataset.from_pandas(df)
data_collector = DataCollatorWithPadding(tokenizer = tokenizer)
custom_map = custom_dataset.map(tokenize_dataset, batched=True)
