import torch
from transformers import pipeline
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import MarianMTModel, MarianTokenizer
def Sentimental_analysis(text):
  Sentiment = pipeline("sentiment-analysis" , model="tabularisai/multilingual-sentiment-analysis")
  return Sentiment(text)
def Ner(text):
  ner = pipeline("ner" , model='dbmdz/bert-large-cased-finetuned-conll03-english' , aggregation_strategy="simple")
  return ner(text)
def Question_Ans( question, text ):
  ques = pipeline("question-answering" , model="deepset/roberta-base-squad2")
  return ques(question = question , context = text)
def text_classification(text , candidate_labels ,multi_label ):
  model = pipeline("zero-shot-classification" )
  return  model(text, candidate_labels=candidate_labels, multi_label=multi_label)
def Summerization(text , max_length , min_length):
  model_name = 'facebook/bart-large-cnn'
  tokenizer = BartTokenizer.from_pretrained(model_name)
  model = BartForConditionalGeneration.from_pretrained(model_name)
  input = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True , padding = True)
  summerizer_idx = model.generate(input, max_length=max_length, min_length=min_length)
  return tokenizer.decode(summerizer_idx[0], skip_special_tokens=True)
def Language_Translation(text , source ,target_language):
  model_name =  f"Helsinki-NLP/opus-mt-{source}-{target_language}"
  tokenizer = MarianTokenizer.from_pretrained(model_name)
  model = MarianMTModel.from_pretrained(model_name)
  input = tokenizer(text, return_tensors="pt" , truncation= True , padding = True)
  translation_idx = model.generate(**input)
  return tokenizer.decode(translation_idx[0] , skip_special_tokens= True)
print("Welcome to the Multi-Stage NLP Processing Engine Application")
print("We have following sevices realed to the NLP  ")
print(" 1. Sentimental Analysis \n 2. Named Entity Recognition \n 3. Question/Answering \n 4. Text Classification \n 5. Summerization \n 6. Language Translation ")
use = input("Do U want to use the service  yes / no  \n")
if use == "yes":
   text = input("Enter the text : \n")
   op = int(input("Which service you want to use (1 - 6) : \n "))
   match op:
      case 1:
        res=Sentimental_analysis(text)
        res=res[0]
        if res['score']>0.5:
          print(f"{res['label']} with score {res['score']}" )
      case 2:
        res = Ner(text)
        for entity in res:
         if entity['score']>0.5:
          print(f"{entity['word']} with score {entity['score']}")
      case 3:
        question = input("Enter the question : \n")
        res =Question_Ans(question , text)
        if res['score']>0.5:
          print(f"{res['answer']} with score {res['score']}")
      case 4:
        labels = []
        candidate_labels = input("Enter the candidate labels seperated by comma : \n")
        labels=[l.strip() for l in candidate_labels.split(",")]
        res=text_classification(text , labels , True)
        for label, score in zip(res['labels'], res['scores']):
         if score > 0.5:
           print(f"{label} with score {score}")
      case 5:
        max = int(input("Enter the max length for summerization : \n"))
        min = int(input("Enter the min length for summerization : \n"))
        print(Summerization(text , max , min))
      case 6:
        print("Which languge U want to traslate form English ")
        print("We have these languages avaliable :\n")
        print("fr : French \n")
        print("ar : Arabic \n")
        print("ur : Urdu \n")
        print("ru : Russian \n")
        print("hi : Hindi \n")
        target = input("Enter the target language : \n")
        print(Language_Translation(text , "en" , target))
else:
  print("Thank U ")




