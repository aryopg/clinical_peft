import argparse
import functools
import os
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv("env/.env")

import huggingface_hub
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import PrefixTuningConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)


def preprocess_dataset(
    dataset,
    tokenizer,
):
    processed_datasets = dataset.map(
        lambda x: tokenizer(
            x["text"],
            max_length=128,
            truncation=True,
        ),
        batched=True,
        num_proc=8,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    # By default, DatasetDict only contains "train" key
    processed_datasets = processed_datasets["train"]

    processed_datasets = processed_datasets.train_test_split(
        test_size=0.1, shuffle=True
    )

    return processed_datasets


peft_hyperparameters = {
    "encoder_hidden_size": 512,
    "num_attention_heads": 12,
    "num_layers": 1,
    "num_transformer_submodules": 1,
    "num_virtual_tokens": 1,
    "prefix_projection": False,
    "token_dim": 768,
}

peft_config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    **peft_hyperparameters,
)

huggingface_hub.login(token=os.getenv("HF_DOWNLOAD_TOKEN", ""))

model = AutoModelForCausalLM.from_pretrained("aryopg/llama-7b")
peft_model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

accelerator = Accelerator(
    gradient_accumulation_steps=2,
    log_with="wandb",
)

dataset = load_dataset("aryopg/mini-mimic-iv", data_files="*.gz")

tokenizer = AutoTokenizer.from_pretrained("aryopg/llama-7b")

dataset = preprocess_dataset(dataset, tokenizer)

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

train_dataloader = DataLoader(
    dataset["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
    pin_memory=True,
)
eval_dataloader = DataLoader(
    dataset["test"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
    pin_memory=True,
)


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# lr scheduler
training_steps = min(len(train_dataloader), 100)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=training_steps,
)

(
    model,
    train_dataloader,
    eval_dataloader,
    optimizer,
    lr_scheduler,
) = accelerator.prepare(
    model,
    train_dataloader,
    eval_dataloader,
    optimizer,
    lr_scheduler,
)


for train_step, batch in enumerate(train_dataloader):
    model.train()
    # Manually remove token type ids
    with accelerator.accumulate(model):
        batch = {k: v for k, v in batch.items() if k != "token_type_ids"}
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        train_loss = loss.detach().float()
        train_ppl = torch.exp(train_loss)
