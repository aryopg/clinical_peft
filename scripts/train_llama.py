import argparse
import datetime
import functools
import os
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv("env/.env")

import huggingface_hub
import torch
import torch.distributed as dist
from datasets import load_dataset
from pynvml import *
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
)

from clinical_peft.configs import Configs
from clinical_peft.utils import common_utils
from clinical_peft.utils.dataset_utils import preprocess_dataset


def argument_parser():
    parser = argparse.ArgumentParser(description="Clinical PEFT")
    parser.add_argument("--config_filepath", type=str, required=True)
    parser.add_argument("--existing_sweep_id", type=str, default="")
    args = parser.parse_args()
    return args


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def main() -> None:
    try:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(seconds=100000000),
        )
    except ValueError:
        pass

    model_path = "aryopg/llama-7b"

    huggingface_hub.login(token=os.getenv("HF_DOWNLOAD_TOKEN", ""))

    args = argument_parser()
    configs = Configs(**common_utils.load_yaml(args.config_filepath))
    dataset = load_dataset("aryopg/mimic-iv", data_files="*.gz")

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")

    dataset = preprocess_dataset(dataset, configs, tokenizer)

    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print_gpu_utilization()

    print("Setup Model")
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )

    print_gpu_utilization()
    training_args = TrainingArguments(
        output_dir="output/",
        do_train=True,
        auto_find_batch_size=True,
        # per_device_train_batch_size=4,
        # per_device_eval_batch_size=4,
        gradient_accumulation_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=1,
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        args=training_args,
    )
    print_gpu_utilization()
    trainer.train()
    print_gpu_utilization()
    trainer.save_state()


if __name__ == "__main__":
    main()
