# importing dataset
import wandb

wandb.login()
import pandas as pd
data = pd.read_csv("train.csv",keep_default_na=False)
print("Imported data: ")
print(data.head())

# For tpu 
import torch_xla.core.xla_model as xm
device = xm.xla_device()
print(f"Running in device: {device}")

#creating a datafunction
datafunc = data[['docstring', 'separatedVariableName']]
comment = datafunc[['docstring']]
variablerecommended = datafunc[['separatedVariableName']]

# converting datafunction into dataset
import datasets
dataset = datasets.Dataset.from_pandas(datafunc)

model_name =f"facebook/bart-large"

# Flatten: 
#   Converting the data into map 
#   or
#   simply converting nested  nested dictionaries are flattened into a single level, with keys representing the path to each leaf node separated by dots. For example, if an example in the dataset has the key-value pair {"a": {"b": 1}}, the flatten function would transform it to {"a.b": 1}
def flatten(example):
    return {
        "input_text": example["docstring"],
        "target_text": example["separatedVariableName"],
    }

# Split samples:
#   Removing any leading or trailing white spaced
def split_samples(example):
    samples = []
    for input_text, target_text in zip(example["input_text"], example["target_text"]):
        if input_text.strip() and target_text.strip():
            samples.append({"input_text": input_text.strip(), "target_text": target_text.strip()})
    return {"text": [sample["input_text"] for sample in samples], "summary": [sample["target_text"] for sample in samples]}

# Mapping each elements using these two functions
print("Mapping each elements using these two functions ...")
dataset = dataset.map(flatten)
dataset = dataset.map(split_samples)

# Importing pretrained models of bart
print("Importing pretrained models ...")
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    Seq2SeqTrainer
)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Seperating Value for training and testing
print("Seperating Value for training and testing ...")
train_data_txt, validation_data_txt = dataset.train_test_split(test_size=0.1).values()


# returns a dictionary with tokenized and processed inputs and labels.
# It also creates a new training dataset (train_data) by applying this function to the train_data_txt dataset using the map() method.
encoder_max_length = 512
decoder_max_length = 64

def batch_tokenize_preprocess(batch, tokenizer, max_source_length, max_target_length):
    source, target = batch["input_text"], batch["target_text"]
    source_tokenized = tokenizer(
        source, padding="max_length", truncation=True, max_length=max_source_length
    )
    target_tokenized = tokenizer(
        target, padding="max_length", truncation=True, max_length=max_target_length
    )

    batch = {k: v for k, v in source_tokenized.items()}
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in l]
        for l in target_tokenized["input_ids"]
    ]
    return batch

# Tokenizing train data
print("Tokenizing train data ...")
train_data = train_data_txt.map(
    lambda batch: batch_tokenize_preprocess(batch, tokenizer, encoder_max_length, decoder_max_length),
    batched=True,
    batch_size=32,
    remove_columns=train_data_txt.column_names,
)

# Tokenizing validation data
print("Tokenizing validation data ...")
validation_data = validation_data_txt.map(
    lambda batch: batch_tokenize_preprocess(batch, tokenizer, encoder_max_length, decoder_max_length),
    batched=True,
    batch_size=32,
    remove_columns= validation_data_txt.column_names,
)

# it takes a batch of examples that have already been tokenized and applies any necessary padding and masking to them.
print("Batching available tokens ...")
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Initializing training arguments
print("Initializing training arguments ...")
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy='steps',
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    gradient_accumulation_steps=16,
    eval_accumulation_steps=28,  
    save_strategy="steps",  
    push_to_hub=True,
    hub_strategy="every_save",
    hub_token="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",
    auto_find_batch_size=True
)

class SaveEachEpochCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_save:
            state.save_model(self.output_dir)

# Initializing trainer
print("Initializing trainer...")
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=validation_data,
    data_collator=data_collator,
    tokenizer=tokenizer,
    callbacks=[SaveEachEpochCallback(output_dir="./results")]
)

import time
print("Training ..")
start = time.time()
trainer.train()
end = time.time()
print(f"Hurrayyy...\n complete in {end-start} seconds")