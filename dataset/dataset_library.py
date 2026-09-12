# !pip install datasets

from datasets import load_dataset
from collections import Counter
import numpy as np
dataset = load_dataset("imdb")
print(dataset)

print("len Train " , len(dataset["train"]))
print("len Test " , len(dataset["test"]))
print("Columns namess", dataset["train"].column_names)
print("Featurs names", dataset['train'].features)
print(dataset["train"][0])
print(dataset["train"][1:5])


lables = dataset['train']['label']
label_counter = Counter(lables)
print(label_counter)
train_texts = dataset['train']['text']
lengths = [len(text.split()) for text in train_texts]

print(f"\nText Length Stats:")
print(f"  Min:     {min(lengths)} words")
print(f"  Max:     {max(lengths)} words")
print(f"  Average: {sum(lengths)//len(lengths)} words")
#filter

positive = dataset['train'].filter(
    lambda x: x['label'] == 1
)
print(f"Positive reviews: {len(positive)}")

negative = dataset['train'].filter(
    lambda x: x['label'] == 0
)
print(f"Negative reviews: {len(negative)}")

short_reviews = dataset['train'].filter(
    lambda x: len(x['text'].split()) < 100
)
print(f"Short reviews: {len(short_reviews)}")

pakistan_reviews = dataset['train'].filter(
    lambda x: 'pakistan' in x['text'].lower()
)
print(f"Pakistan mentions: {len(pakistan_reviews)}")
