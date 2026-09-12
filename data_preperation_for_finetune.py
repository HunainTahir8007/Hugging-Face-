from transformers import AutoTokenizer
from datasets import load_dataset

mod = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(mod)
dataset = load_dataset("imdb")
print(dataset.shape)
tokenizer.vocab_size
print(dataset['train'][0]['text'][:100])
print(dataset['train'][10:20])
sample = dataset['train'][12]["text"]
tokenized_output = tokenizer(sample, add_special_tokens=True, padding=True, truncation=True, return_tensors='pt')
print(tokenized_output['input_ids'])
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

tokeized_dataset = dataset.map(tokenize_function, batched=True)
before_map = dataset['train'].column_names
after_map = tokeized_dataset['train'].column_names
print(before_map)
print(after_map)
