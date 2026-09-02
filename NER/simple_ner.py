from transformers import pipeline
import torch

ner = pipeline("ner", )
txt = "Elon Musk founded Tesla in California"
res= ner(txt)
res
# ner with agrigation
neer = pipeline("ner" , aggregation_strategy="simple")

neer(txt)
sentences = [
    "Elon Musk founded Tesla in California",
    "Bill Gates created Microsoft in Seattle in 1975",
    "Imran Khan was born in Lahore Pakistan",
    "Mark Zuckerberg built Facebook at Harvard University",
    "Apple was founded by Steve Jobs in Cupertino",
    "Cristiano Ronaldo plays for Al Nassr in Saudi Arabia",
    "Ahmed Khan studied at FAST University in Lahore"]
for i in sentences:
  res = neer(i)
  for entity in res:
    print(f"{entity["word"]} -----> {entity["entity_group"]}")
