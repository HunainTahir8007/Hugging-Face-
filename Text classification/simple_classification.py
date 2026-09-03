import torch
from transformers import pipeline
txt_classifier = pipeline("zero-shot-classification")
text = "i am learning the deep learning to get hired"
labels=["education", "technology", "sports", "finance"]
result=txt_classifier(text,candidate_labels=labels)
print(text)
print(result)
for label , score in zip(result["labels"], result["scores"]):
   bar = "-" * int(score * 20)  
   print(f"{label:15} {bar} {score*100:.2f}%")
