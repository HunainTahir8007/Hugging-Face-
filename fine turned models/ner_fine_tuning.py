import transformers
from transformers import AutoTokenizer , AutoModelForTokenClassification
from datasets import load_dataset
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm.auto import tqdm
from seqeval.metrics import f1_score as ner_f1
from seqeval.metrics import classification_report

!pip install seqeval
dataset = load_dataset("wikiann", "en")

print(dataset)
print(len(dataset['train']))
print(len(dataset['test']))
samp = dataset['train'][0]
print(samp['tokens'])
print(samp['ner_tags'])
print(samp['spans'])
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenizer_dataset(dataset):
    tokenized_inputs = tokenizer(dataset['tokens'], truncation=True, is_split_into_words=True , padding='max_length' , max_length=128)

    labels= []
    for i, label in enumerate(dataset['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs['labels'] = labels
    return tokenized_inputs



train_tokenized = dataset['train'].map(
    tokenizer_dataset,
    batched=True,
    remove_columns=dataset['train'].column_names
)
val_tokenized = dataset['test'].map(
    tokenizer_dataset,
    batched=True,
    remove_columns=dataset['test'].column_names
)
train_tokenized.set_format("torch")
val_tokenized.set_format("torch")

print(f"Train: {len(train_tokenized)}")
print(f"Val:   {len(val_tokenized)}")
print(f"Columns: {train_tokenized.column_names}")
train_loader = DataLoader(
    train_tokenized,
    batch_size=64,
    shuffle=True
)
val_loader = DataLoader(
    val_tokenized,
    batch_size=64,
    shuffle=False
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches:   {len(val_loader)}")
label_names = dataset['train'].features['ner_tags'].feature.names
num_labels  = len(label_names)

id2label = {i: label for i, label in enumerate(label_names)}
label2id = {label: i for i, label in enumerate(label_names)}
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = model.to(device)
print(f"Num labels: {num_labels}")
print(f"Device:     {device}")
print(f"Labels:     {label_names}")
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5 , weight_decay=0.01)
num_epochs        = 3
num_training_steps = num_epochs * len(train_loader)
num_warmup_steps  = 100
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)
# do not use loss because the loss is measured automatically

train_tokenized.column_names
# training
def training(model , loder , optimizer , scheduler , device):
  model.train()
  total_loss = 0
  num_batches = 0
  for batch_idx , batch in enumerate(loder):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    outputs = model(input_ids , attention_mask=attention_mask , labels=labels)
    loss = outputs.loss
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    total_loss += loss.item()
    num_batches += 1
    if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / num_batches
            print(f"  Batch {batch_idx+1}/{len(loder)} | Loss: {avg_loss:.4f}")
  return total_loss / num_batches

def evaluate(model, loader, device, label_names):
    model.eval()
    total_loss = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss   = outputs.loss
            logits = outputs.logits
            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=-1)
            preds_np    = predictions.cpu().numpy()
            labels_np   = labels.cpu().numpy()

            for pred_seq, label_seq in zip(preds_np, labels_np):
                pred_tags  = []
                label_tags = []

                for pred, label in zip(pred_seq, label_seq):
                    if label != -100:
                        pred = min(pred, len(label_names) - 1)
                        pred_tags.append(label_names[pred])
                        label_tags.append(label_names[label])

                if len(pred_tags) > 0:
                    all_preds.append(pred_tags)
                    all_labels.append(label_tags)

    f1       = ner_f1(all_labels, all_preds)
    avg_loss = total_loss / len(loader)

    return avg_loss, f1, all_preds, all_labels
# complete loop
print("-"*50)
print("  NER FINE-TUNING WITH  MANUAL TRAINING LOOP")   # wothout trainer
print("="*55)
print(f"Epochs:     {num_epochs}")
print(f"Train size: {len(train_tokenized)}")
print(f"Val size:   {len(val_tokenized)}")
print(f"Device:     {device}")
print("="*55)
best_f1    = 0
best_epoch = 0
for epoch in range(num_epochs):
    print(f"\n{'─'*55}")
    print(f"EPOCH {epoch+1}/{num_epochs}")
    print(f"{'─'*55}")
    train_loss = training(model , train_loader , optimizer , scheduler , device)
    print(f"Train Loss: {train_loss:.4f}")

    # model prformace
    val_loss, val_f1, preds, labels = evaluate(
        model, val_loader, device, label_names
    )
    print(f"Val Loss:   {val_loss:.4f}")
    print(f"Val F1:     {val_f1*100:.2f}%")
    if val_f1 > best_f1:
        best_f1    = val_f1
        best_epoch = epoch + 1
        model.save_pretrained("./ner-bert-best")
        tokenizer.save_pretrained("./ner-bert-best")


from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
import torch

ner_model     = AutoModelForTokenClassification.from_pretrained("./ner-bert-best")
ner_tokenizer = AutoTokenizer.from_pretrained("./ner-bert-best")
ner_model     = ner_model.to(device)
ner_model.eval()

def predict_ner(text):

    inputs = ner_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = ner_model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=-1)
    predictions = predictions[0].cpu().numpy()
    tokens = ner_tokenizer.convert_ids_to_tokens(
        inputs['input_ids'][0].cpu().numpy()
    )

    print(f"\nText: {text}")
    print("─"*50)
    print(f"{'Token':20} {'Label':10}")
    print("─"*50)

    for token, pred in zip(tokens, predictions):
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            continue
        label = ner_model.config.id2label[pred]
        if label != 'O':
            print(f"{token:20} {label:10}")


predict_ner("Elon Musk founded Tesla in California")
predict_ner("Imran Khan was born in Lahore Pakistan")
predict_ner("Microsoft was created by Bill Gates in Seattle")
predict_ner("I am a student at FAST University in Lahore")
import shutil
from google.colab import files

shutil.make_archive(
    "ner-bert-best",
    "zip",
    "./ner-bert-best"
)

# Download
files.download("ner-bert-best.zip")
print("Download started ")
