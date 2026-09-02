from transformers import pipeline
classifier = pipeline("sentiment-analysis")
resullt= classifier("i love to code")
print(resullt)
resullt= classifier("i hate to code")
print(resullt)
result = classifier(["i love to code" , "i hate to code"])
print(resullt)
print(classifier("The new update is fantastic and doubled my productivity!"))
print(classifier("The product arrived on Tuesday as expected"))
print(classifier("The camera quality is great, but the battery life is poor"))
# MULTIPLE SENTENCES
sentences = [
    "I love machine learning",
    "This is the worst day ever",
    "HuggingFace is amazing",
    "I hate bugs in my code",
    "Pakistan is a beautiful country",
    "pakistan is not a good country"
]
results = classifier(sentences)
for sentence , resullt in zip(sentences, results):
  print(f"text {sentence}")
  print(f"Label: {resullt['label']} | Score: {resullt['score']:.4f}")
  print("─" * 50)
Using the coustom model
classifier = pipeline("text-classification", model = "tabularisai/multilingual-sentiment-analysis")

print(classifier(["I absolutely love the new design of this app!", "The customer service was disappointing.", "The weather is fine, nothing special."]))
def text_class(text):
  result = classifier(text)
  label = result[0]["label"]
  score = result[0]["score"] *100

  emoji = "👍"

  print(f"text  {text}")
  print(f"label {label} {emoji}")
  print(f"score {score:.2f}")
  print("─" * 50)
text_class("I built a Transformer from scratch!")
text_class("I am tired of financial problems")
text_class("Tomorrow will be better")
text_class("HuggingFace is going to get me my first dollar")