from transformers import pipeline
ans = pipeline("question-answering")

text = '''Pakistan is a country in South Asia.
Islamabad is the capital of Pakistan.
Lahore is the cultural capital and largest city of Punjab.
Pakistan was founded on 14 August 1947.
Muhammad Ali Jinnah was the founder of Pakistan.'''

question = "What is the country whose capital is Islamabad?"
result = ans(question = question, context = text)
print(result)
