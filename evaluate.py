import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "deberta_model"            # change to match your model dir
test_path = 'data/test_essays.csv'      # change to match your test data file

tokenizer = AutoTokenizer.from_pretrained(model_path)                   # loading tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_path)  # loading model

print('Done loading tokenizer & model')

# function for predicting batch of samples
def predict_proba(texts, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        # Make sure output is always a list, even if single item
        if probs.ndim == 0:
            probs = [probs.item()]
        elif probs.ndim == 1:
            probs = probs.tolist()

        all_probs.extend(probs)

    return all_probs

df = pd.read_csv(test_path)     # must have 'text' column
texts = df["text"].tolist()     # to match the 'predict_proba' function requirements

print('done loading data')

probs = predict_proba(texts)    # predict samples

# format & save final submission dataset
df["generated"] = probs                    
df = df[["id", "generated"]]
df.to_csv("submission.csv", index=False)

print('Done!!')