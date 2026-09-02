import torch
from transformers import pipeline

ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
sentence = input("Enter the text of resume \n")
resume_analyzer = ner(sentence)

for result in resume_analyzer:
      if result['score'] > 0.70:
        print(f"Class --> {result['entity_group']} Name --> {result['word']} with socre -->  {result['score']}")

