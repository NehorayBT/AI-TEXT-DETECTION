
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import os
import evaluate

################################
### LOADING & SPLITTING DATA ###
################################

data_path = 'train_essays.csv'  # change to where you store your training data
output_dir = 'deberta_model'    # change to desired output folder

# reading data
essay_df = pd.read_csv(data_path)
essay_df = essay_df[["text", "generated"]].dropna()

# splitting data (as mentioned, we want to train on as much samples as possible,
# and dont care much for the valid score, and so we make a very small validation size)
train_df, valid_df = train_test_split(essay_df, test_size=0.005, random_state=42)
print('Done loading data')

################################
######## PREPARING DATA ########
################################

# loading the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/deberta-v3-large",
        padding_side='left',
        truncation_side='left',
    )

# a function for preprocessing (if needed)
def preprocess_function(df):
    # TODO: Add Preprocessing!
    return df

# a function for tokenizing the data
def tokenize_function(ds):
    return tokenizer(
        ds['text'],
        padding='max_length',
        truncation=True,
        max_length=512,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

# pre-processes, tokenizes and formatting the data
def get_preprocessed_dataset(df):
    df = preprocess_function(df)  # Apply text modifications
    df.rename(columns={"generated": "labels"}, inplace=True)
    ds = Dataset.from_pandas(df)  # Convert DataFrame to HF Dataset

    # Tokenize
    ds = ds.map(tokenize_function, batched=True)

    # ds.set_format(type=None, columns=['input_ids', 'attention_mask', 'labels'])
    return ds

# executing the above on both datasets
train_ds = get_preprocessed_dataset(train_df)
valid_ds = get_preprocessed_dataset(valid_df)
print('Done preprocessing')

################################
######### LOADING MODEL ########
################################

model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-large", num_labels=1)

################################
########### TRAINING ###########
################################

# loading evaluation method
roc_auc = evaluate.load("roc_auc")

# compute metrics for evaluation
def compute_metrics(pred):
    logits, labels = pred
    probs = 1 / (1 + np.exp(-logits))  # Sigmoid for binary output
    return roc_auc.compute(prediction_scores=probs, references=labels)

# setting up training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.001,
    fp16=True,
    report_to='none',
)

# initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the last model (we dont care for valid score)
trainer.save_model(os.path.join(output_dir, "last_model"))

print('Done')

