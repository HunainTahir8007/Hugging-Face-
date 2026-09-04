from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer
text = "I love to learn the deep learning"
token = tokenizer.tokenize(text)
print(token)
token_to_ids = tokenizer.convert_tokens_to_ids(token)  # convert tokens to idx
print(token_to_ids)
#  encode (tokenize + convert + add special tokens)
input_ids = tokenizer.encode(text)
print(input_ids)
print(tokenizer.special_tokens_map)
print(f"CLS ID:  {tokenizer.cls_token_id}")
print(f"SEP ID:  {tokenizer.sep_token_id}")
print(f"PAD ID:  {tokenizer.pad_token_id}")
print(f"UNK ID:  {tokenizer.unk_token_id}")
print(f"MASK ID: {tokenizer.mask_token_id}")
# full pipeline
text = "I love to learn the deep learning with pytorch framework"
tokenize = tokenizer(text , retrun_tensors="pt", max_length=512 , padding = True , truncation = True , add_special_tokens=True , return_attention_mask=True , return_token_type_ids=True)
print(tokenize)
# padding concept
sentences = [
    "I love ML",
    "I love deep learning with PyTorch"
]
res = tokenizer(sentences, return_tensors="pt", padding=True)
print(res["input_ids"])
print(res['attention_mask'])
