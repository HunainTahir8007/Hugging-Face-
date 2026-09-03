import torch
from transformers import pipeline
txt = pipeline("zero-shot-classification")

# single
text = "i hope i will do something for my beloved country"
labels = ['happiness','working', 'technology', 'desire']
result = txt(text, candidate_labels=labels)
result
# multi class labels
text = "i am working hard to learn the deep learning to secure my future this is my desire"
lables = ['happiness','working', 'carrier', 'technology', 'dream','struggling']
result = txt(text, candidate_labels=lables, multi_label=True)
print(f"label  {result['labels']} ---- score {result['scores']}")

# Simple news detector by text classification
news = input("Enter the news")
labels = [
        "real news",
        "fake news",
        "satire",
        "clickbait",
        "opinion"
    ]
result = txt(news, candidate_labels=labels, multi_label=True)
for labels , scores in zip(result['labels'],result['scores']):
        print(f"  {labels} ---> {scores}")
