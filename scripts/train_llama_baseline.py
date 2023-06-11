import argparse
import json
import math
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import datasets
import evaluate
import huggingface_hub
import numpy as np
import torch
import torch.nn.functional as F
import tqdm.auto as tqdm
import transformers
from accelerate import Accelerator
from accelerate.tracking import WandBTracker
from datasets import load_dataset
from evaluate import EvaluationModule
from peft import PeftConfig, PeftModel, get_peft_model
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    LlamaForCausalLM,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)

from clinical_peft.configs import Configs, TaskType
from clinical_peft.constants import (
    LABELS_MAP,
    LLAMA_SPECIAL_CHARACTER_IDS,
    LLAMA_SPECIAL_CHARACTERS,
)
from clinical_peft.utils import common_utils
from clinical_peft.utils.dataset_utils import preprocess_dataset


def argument_parser():
    parser = argparse.ArgumentParser(description="Clinical PEFT")
    parser.add_argument("--config_filepath", type=str, required=True)
    parser.add_argument("--existing_sweep_id", type=str, default="")
    args = parser.parse_args()
    return args


def main() -> None:
    args = argument_parser()
    configs = Configs(**common_utils.load_yaml(args.config_filepath))
    huggingface_hub.login(token=os.getenv("HF_DOWNLOAD_TOKEN", ""))

    print("Setup Data")
    dataset = load_dataset(configs.training_configs.dataset_paths[0], data_files="*.gz")

    # Load Tokenizer
    if any(
        k in configs.model_configs.model_name_or_path for k in ("gpt", "opt", "bloom")
    ):
        padding_side = "left"
    else:
        padding_side = "right"
    tokenizer: AutoTokenizer.from_pretrained(
        configs.model_configs.model_name_or_path, padding_side=padding_side
    )

    dataset = preprocess_dataset(dataset, configs, tokenizer)

    print("Setup Model")
    model = LlamaForCausalLM.from_pretrained(
        configs.model_configs.model_name_or_path, return_dict=True
    )

    if "PMC_LLAMA" in configs.model_configs.model_name_or_path:
        for special_char in ["unk", "bos", "pad", "eos"]:
            setattr(
                tokenizer,
                special_char + "_token",
                LLAMA_SPECIAL_CHARACTERS[special_char],
            )
            setattr(
                tokenizer,
                special_char + "_token_id",
                LLAMA_SPECIAL_CHARACTER_IDS[special_char],
            )

    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if configs.model_configs.task_type == TaskType.causal_lm:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    elif configs.model_configs.task_type == TaskType.seq_cls:
        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding="longest",
            return_tensors="pt",
        )

    train_dataloader = DataLoader(
        dataset["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=configs.training_configs.batch_size,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        dataset["test"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=configs.training_configs.batch_size,
        pin_memory=True,
    )

    training_args = TrainingArguments(
        output_dir="llama",
        evaluation_strategy="epoch",
        per_device_train_batch_size=4,
        do_train=True,
        do_eval=True,
        save_strategy="epoch",
        gradient_accumulation_steps=10,
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=True,
        push_to_hub=True,
        hub_token=os.getenv("HF_UPLOAD_TOKEN"),
        hub_private_repo=True,
        fsdp="full_shard auto_wrap",
        fsdp_transformer_layer_cls_to_wrap="LlamaDecoderLayer",
    )

    huggingface_hub.login(token=os.getenv("HF_UPLOAD_TOKEN", ""))
    trainer = Trainer(
        model=model,
        train_dataset=train_dataloader,
        eval_dataset=test_dataloader,
        args=training_args,
    )
    # trainer.train(resume_from_checkpoint=True)
    trainer.train()
    trainer.save_state()


if __name__ == "__main__":
    main()
