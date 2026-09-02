import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_name = "tabularisai/multilingual-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def perdict_sentiment(text_input):
  inputs = tokenizer(text_input , return_tensors="pt", max_length=512 , truncation=True , padding =True)
  with torch.no_grad():
    outputs= model(**inputs)
    logits = torch.nn.functional.softmax(outputs.logits , dim=-1) # Corrected: use outputs.logits
    sentiment_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
    return [sentiment_map[p] for p in torch.argmax(logits, dim=-1).tolist()]

texts = [
    "I absolutely love the new design of this app!", "The customer service was disappointing.", "The weather is fine, nothing special.",
    "这家餐厅的菜味道非常棒！", "我对他的回答很失望。", "天气今天一般。",
    "¡Me encanta cómo quedó la decoración!", "El servicio fue terrible y muy lento.", "El libro estuvo más o menos.",
    "الخدمة في هذا الفندق رائعة جدًا!", "لم يعجبني الطعام في هذا المطعم.", "كانت الرحلة عادية。",
    "Мені дуже сподобалася ця вистава!", "Обслуговування було жахливим.", "Книга була посередньою。",

    "यह जगह सच में अद्भुत है!", "यह अनुभव बहुत खराब था।", "फिल्म ठीक-ठाक थी।",

    "এখানকার পরিবেশ অসাধারণ!", "সেবার মান একেবারেই খারাপ।", "খাবারটা মোটামুটি ছিল।",

    "Este livro é fantástico! Eu aprendi muitas coisas novas e inspiradoras.",
    "Não gostei do produto, veio quebrado.", "O filme foi ok, nada de especial.",

    "このレストランの料理は本当に美味しいです！", "このホテルのサービスはがっかりしました。", "天気はまあまあです。",

    "Я в восторге от этого нового гаджета!", "Этот сервис оставил у меня только разочарование.", "Встреча была обычной, ничего особенного.",

    "J'adore ce restaurant, c'est excellent !", "L'attente était trop longue et frustrante.", "Le film était moyen, sans plus.",

    "Bu otelin manzarasına bayıldım!", "Ürün tam bir hayal kırıklığıydı.", "Konser fena değildi, ortalamaydı.",

    "Adoro questo posto, è fantastico!", "Il servizio clienti è stato pessimo.", "La cena era nella media.",

    "Uwielbiam tę restaurację, jedzenie jest świetne!", "Obsługa klienta była rozczarowująca.", "Pogoda jest w porządku, nic szczególnego.",

    "Ang ganda ng lugar na ito, sobrang aliwalas!", "Hindi maganda ang serbisyo nila dito.", "Maayos lang ang palabas, walang espesyal.",

    "Ik ben echt blij met mijn nieuwe aankoop!", "De klantenservice was echt slecht.", "De presentatie was gewoon oké, niet bijzonder.",
    "Saya suka makanan di sini, sangat sedap!", "Pengalaman ini sangat mengecewakan.", "Hari ini cuacanya biasa sahaja.",
    "이 가게의 케이크는 정말 맛있어요!", "서비스가 너무 별로였어요.", "날씨가 그저 그렇네요.",
    "Ich find dä Service i de Beiz mega guet!", "Däs Esä het mir nöd gfalle.", "D Wätter hüt isch so naja."
]

for text , sentiment in zip(texts, perdict_sentiment(texts)):
   print(f"Text: {text}\nSentiment: {sentiment}\n")
