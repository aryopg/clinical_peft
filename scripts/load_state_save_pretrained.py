import os

import huggingface_hub
import torch
from accelerate import Accelerator
from datasets import DatasetDict, load_dataset
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from clinical_peft.configs import Configs
from clinical_peft.trainer import setup_data_loader
from clinical_peft.utils import common_utils
from clinical_peft.utils.dataset_utils import preprocess_dataset

huggingface_hub.login(token="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")

accelerator = Accelerator(
    gradient_accumulation_steps=64,
)

model = AutoModelForCausalLM.from_pretrained(
    "aryopg/llama-7b",
    return_dict=True,
)

# optimizer
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

optimizer = AdamW(
    params=[param for param in model.parameters() if param.requires_grad],
    lr=0.0003,
)

# lr scheduler
warmup_steps_ratio = 0.06

num_training_steps = 1000000
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=warmup_steps_ratio,
    num_training_steps=num_training_steps,
)

configs = Configs(
    **common_utils.load_yaml(
        "configs/mimic_pretrain_hpo_configs/llama_baseline_low_lr.yaml"
    )
)
max_batch_size = 4
padding_side = "right"
dataset = load_dataset("aryopg/mini-mimic-iv", data_files="*.gz")
tokenizer = AutoTokenizer.from_pretrained(
    configs.model_configs.model_name_or_path, padding_side=padding_side
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
dataset = preprocess_dataset(dataset, configs, tokenizer)
train_dataloader, val_dataloader, test_dataloader = setup_data_loader(
    configs, tokenizer, dataset, max_batch_size
)

(
    model,
    optimizer,
    lr_scheduler,
    train_dataloader,
    test_dataloader,
) = accelerator.prepare(
    model, optimizer, lr_scheduler, train_dataloader, test_dataloader
)

# print("Loading state")
# accelerator.load_state("outputs/2023_09_25__16_41_23/checkpoint")
# print("Loaded state")

# accelerator.save_model(model, "outputs/clinical_llama", max_shard_size="10GB", safe_serialization=True)
print("Unwrap model")
unwrapped_model = accelerator.unwrap_model(model)
print("Saving pretrained model")
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

fp32_model = load_state_dict_from_zero_checkpoint(
    unwrapped_model, "outputs/2023_09_25__16_41_23/checkpoint"
)
fp32_model.save_pretrained(
    "outputs/clinical_llama_fp32",
    is_main_process=accelerator.is_main_process,
    save_function=accelerator.save,
    state_dict=accelerator.get_state_dict(model),
)
print("Done")
