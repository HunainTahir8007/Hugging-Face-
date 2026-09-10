!pip install transformers accelerate bitsandbytes peft datasets trl gradio sentencepiece
from google.colab import files
import zipfile
import os

print("Select your mistral-hunain-gpt.zip file")
uploaded = files.upload()

print("Extracting...")
with zipfile.ZipFile("mistral-hunain-gpt.zip", "r") as zip_ref:
    zip_ref.extractall("./mistral-hunain-gpt")

print("Extracted files:")
for f in os.listdir("./mistral-hunain-gpt"):
    print(f"  ✅ {f}")
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import PeftModel
import torch
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
model = PeftModel.from_pretrained(
    base_model,
    "./mistral-hunain-gpt"
)


model.eval()
device = next(model.parameters()).device
print(f"Model ready for the testing on data")
print(f"Device: {device}")
SYSTEM_PROMPT = """You are Hunain Tahir's Personal AI Assistant.

About Hunain Tahir:
- Name: Hunain Tahir
- Role: Machine Learning Engineer
- University: Government College University Faisalabad
- City: Mailsi Punjab Pakistan
- Skills: Machine Learning, Deep Learning, PyTorch, HuggingFace, CNN, RNN, Transformers, Computer Vision
- Projects:
  1. GPT Transformer from scratch (33% accuracy on Shakespeare)
  2. Plant Disease Detector (83.4% accuracy, 38 classes, 67K images)
  3. Vehicle Classifier (CNN: 25-53%, Transfer Learning: 85-88%)
  4. Food Classifier EfficientNet_B0 (89% accuracy: pizza, steak, sushi)
  5. Laptop Price Predictor (XGBoost, SVR, Linear Regression)
- GitHub: github.com/HunainTahir8007
- LinkedIn: linkedin.com/in/hunain-tahir-08b7a8370
- Goal: Senior Machine Learning Engineer

RULES:
- If asked about Hunain (name, skills, projects, GitHub, etc.) → answer from above info
- If asked general questions (cooking, science, jokes, ML concepts) → answer helpfully as AI
- Be friendly and professional
"""

PERSONAL_KEYWORDS = [
    "hunain","who are you","about you","tell me you",  "project","skill","github","linkedin",
    "university","goal","built","created","plant","transformer",
    "vehicle","food", "laptop","classifier","detector","your skill",    "your project",  "your github",  "your goal",
    "your name", "your university"
]

PERSONAL_QA = {
    "name":       "I am Hunain Tahir's AI Assistant. Hunain is a Machine Learning Engineer from Pakistan.",
    "who":        "Hunain Tahir is a Machine Learning Engineer studying at Government College University Faisalabad Pakistan.",
    "skill":      "Hunain specializes in Machine Learning, Deep Learning, PyTorch, HuggingFace, CNN, RNN, Transformers, and Computer Vision.",
    "project":    "Hunain built GPT Transformer from scratch, Plant Disease Detector 99.4%, Vehicle Classifier, Food Classifier, and Laptop Price Predictor.",
    "github":     "Find all projects at github.com/HunainTahir8007",
    "linkedin":   "Connect at linkedin.com/in/hunain-tahir-08b7a8370",
    "university": "Hunain studies Software Engineering at Government College University Faisalabad.",
    "goal":       "Hunain's goal is to become a Senior Machine Learning Engineer.",
    "city":       "Hunain is from Mailsi Punjab Pakistan.",
    "transformer":"Hunain built a GPT Transformer from scratch using PyTorch achieving 33% accuracy on Shakespeare text.",
    "plant":      "Hunain built a Plant Disease Detector classifying 38 diseases with 99.4% accuracy.",
    "food":       "Hunain built a Food Classifier using EfficientNet_B0 achieving 89% accuracy.",
    "vehicle":    "Hunain built a Vehicle Classifier comparing CNN 25-53% vs Transfer Learning 85-88%.",
    "laptop":     "Hunain built a Laptop Price Predictor using XGBoost SVR and Linear Regression.",
}

def is_personal_question(question):
    q = question.lower()
    return any(keyword in q for keyword in PERSONAL_KEYWORDS)

def get_personal_answer(question):
    q = question.lower()
    if any(w in q for w in ["name", "who are", "yourself"]):
        return PERSONAL_QA["name"]
    elif any(w in q for w in ["who is", "hunain"]):
        return PERSONAL_QA["who"]
    elif any(w in q for w in ["skill", "know", "expert"]):
        return PERSONAL_QA["skill"]
    elif any(w in q for w in ["project", "built", "made"]):
        return PERSONAL_QA["project"]
    elif "github" in q:
        return PERSONAL_QA["github"]
    elif "linkedin" in q:
        return PERSONAL_QA["linkedin"]
    elif any(w in q for w in ["university", "study", "college"]):
        return PERSONAL_QA["university"]
    elif any(w in q for w in ["goal", "aim", "future"]):
        return PERSONAL_QA["goal"]
    elif any(w in q for w in ["city", "from", "where"]):
        return PERSONAL_QA["city"]
    elif "transformer" in q:
        return PERSONAL_QA["transformer"]
    elif "plant" in q:
        return PERSONAL_QA["plant"]
    elif "food" in q:
        return PERSONAL_QA["food"]
    elif "vehicle" in q:
        return PERSONAL_QA["vehicle"]
    elif "laptop" in q:
        return PERSONAL_QA["laptop"]
    else:
        return PERSONAL_QA["who"]

GENERAL_PROMPT = """You are a helpful AI assistant.
Answer the question directly and helpfully.
Give practical useful information.
Do not mention any personal information."""

def get_response(message, history=[]):

    if is_personal_question(message):
        return get_personal_answer(message)

    else:
        prompt = f"<s>[INST] <<SYS>>\n{GENERAL_PROMPT}\n<</SYS>>\n\n"

        for human, assistant in history:
            prompt += f"{human} [/INST] {assistant} </s><s>[INST] "

        prompt += f"{message} [/INST]"

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                inputs['input_ids'],
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(
            output[0][inputs['input_ids'].shape[-1]:],
            skip_special_tokens=True
        )

        return response.strip()
test_questions = [

    "Who is Hunain Tahir?",
    "What are your skills?",
    "What is your GitHub?",

    "Give me cooking tips",
    "What is the capital of Pakistan?",
    "How to reduce stress?",
    "Tell me a joke",
    "What is machine learning?",
    "How to wake up early?",
]

print("="*55)
print("ROUTING TEST")
print("="*55)

for q in test_questions:
    response = get_response(q)
    route = "PERSONAL" if is_personal_question(q) else "GENERAL"
    print(f"\n[{route}] Q: {q}")
    print(f"         A: {response}")
    print("─"*55)

import gradio as gr

def respond(message, history):
    response = get_response(message, history)
    return response

app = gr.ChatInterface(
    fn=respond,
    title="🤖 NOX — Personal AI Assistant",
    description="""
     Hi! I am Hunain's Personal AI Assistant.\n
     ----------Trained on-----------\n
    ||| General question or any othet\n
    ||| Tech Knowledge\n
    ||| Normal conversation\n
    ||| About me\n


           """,
    examples=[
        "Ways to reduce stress?",
        "Suggest me things which i do in spare time?",
        "Who is thr Prime Minister of Pakistan?",
    ],
    theme=gr.themes.Soft()
)

app.launch(share=True)
