import argparse
import os

import evaluate
import torch
from datasets import load_dataset
from peft import (
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

batch_size = 32
model_name_or_path = "emilyalsentzer/Bio_ClinicalBERT"
dataset_name = "aryopg/trec-mortality"
peft_type = PeftType.LORA
device = "cuda"
num_epochs = 20
max_length = 128

peft_config = LoraConfig(
    task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1
)
lr = 3e-4

if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

datasets = load_dataset(dataset_name)
metric = evaluate.load("f1")

tokenized_datasets = datasets.map(
    lambda x: tokenizer(
        x["text"],
        max_length=max_length,
        truncation=True,
    ),
    batched=True,
    num_proc=8,
    remove_columns=[
        column_name
        for column_name in datasets["train"].column_names
        if not column_name == "label"
    ],
    load_from_cache_file=True,
    desc="Running tokenizer on dataset",
)

# We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
# transformers library
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")


def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")


# Instantiate dataloaders.
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=collate_fn,
    batch_size=batch_size,
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"],
    shuffle=False,
    collate_fn=collate_fn,
    batch_size=batch_size,
)


model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path, return_dict=True
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


optimizer = AdamW(params=model.parameters(), lr=lr)

# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs),
    num_training_steps=(len(train_dataloader) * num_epochs),
)

model.to(device)
for epoch in range(num_epochs):
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items() if k != "token_type_ids"}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items() if k != "token_type_ids"}
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )
    print(outputs)
    print(batch["labels"])

    eval_metric = metric.compute(average="micro")
    print(f"epoch {epoch}:", eval_metric)
