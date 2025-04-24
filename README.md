# AI-Generated Essay Detector

This repository contains code to train a DeBERTa-v3-Large model to detect AI-generated essays.

## Included Files

- `data/train_essays.csv`: Contains a small demo sample from the AI-MIX-V16 dataset
- `data/test_essays.csv`: Contains the demo test set from the **LLM - Detect AI Generated Text** Kaggle competition

## Model Files

The trained model files are not included in this repository due to their size. To download our pre-trained model:

1. Download the model files from [Kaggle: DAI-Ver-11-Model](https://www.kaggle.com/models/nehoraybentzvi/dai-ver-11-model/Other/default/1)
2. Extract the downloaded files to a directory named `deberta_model` in the root of this repository

## Data Requirements

The repository includes a small demo dataset in the `data/train_essays.csv` file that you can use to test the code.

## Replication
To replicate our exact trained model:

1. Download the complete dataset from [Kaggle: AI Mix v16](https://www.kaggle.com/datasets/conjuring92/ai-mix-v16)
2. Place the dataset file in the `data` directory and name it `train_essays.csv` (replacing the demo file)
3. Run `python train.py` without changing any of the default settings


## Installation

Install the required dependencies:

```bash
pip install pandas scikit-learn transformers datasets evaluate numpy torch
```

## Running the Code

### Training
To train the model, simply run:

```bash
python train.py
```

### Evaluation
To evaluate the trained model on test data, run:

```bash
python evaluate.py
```

This will generate a `submission.csv` file with the model's predictions.

## Customization

### Training Parameters
You can customize various parameters by editing the `train.py` file:

### Data Path and Output Directory
Located near the beginning of the file (around line 13-14):

```python
data_path = 'data/train_essays.csv'  # change to where you store your training data
output_dir = 'deberta_model'    # change to desired output folder
```

### Validation Split Size
Located in the data splitting section (around line 19):

```python
train_df, valid_df = train_test_split(essay_df, test_size=0.005, random_state=42)
```
Change `test_size=0.005` to your preferred validation set size.

### Training Arguments
Located in the training section (around line 90-101):

```python
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
```

You can modify parameters like:
- `learning_rate`
- `per_device_train_batch_size` and `per_device_eval_batch_size` (decrease if you have GPU memory issues)
- `num_train_epochs` (increase for potentially better performance)
- `weight_decay` and other hyperparameters

## Hardware Requirements

This training script uses fp16 precision and is optimized to run on a GPU. Training transformer models like DeBERTa-v3-Large is resource-intensive and may take considerable time even on high-end hardware.

### Evaluation Parameters
You can customize the evaluation parameters by editing the `evaluate.py` file:

```python
model_path = "deberta_model"            # change to match your model dir
test_path = 'data/test_essays.csv'      # change to match your test data file
```

- `model_path`: Path to the directory containing your trained model
- `test_path`: Path to your test data CSV file (must have 'text' and 'id' columns)

## Output

### Training Output
The trained model will be saved in the specified `output_dir` folder. Checkpoints will be saved after each epoch, with the final model saved in a "last_model" subfolder.

### Evaluation Output
The evaluation script will generate a `submission.csv` file containing the model's predictions. This file will include two columns:
- `id`: The ID from the test data
- `generated`: The probability score that the essay was AI-generated
