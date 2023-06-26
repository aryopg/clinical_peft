import argparse
import datetime
import functools
import os
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv("env/.env")

import huggingface_hub
from datasets import load_dataset
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


def main() -> None:
    max_batch_size = 32
    model_path = "aryopg/llama-7b"

    huggingface_hub.login(token=os.getenv("HF_DOWNLOAD_TOKEN", ""))

    args = argument_parser()
    configs = Configs(**common_utils.load_yaml(args.config_filepath))
    dataset = load_dataset("aryopg/mini-mimic-iv", data_files="*.gz")

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")

    dataset = preprocess_dataset(dataset, configs, tokenizer)

    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # train_dataloader = DataLoader(
    #     dataset["train"],
    #     shuffle=True,
    #     collate_fn=data_collator,
    #     batch_size=max_batch_size,
    #     pin_memory=True,
    # )
    # test_dataloader = DataLoader(
    #     dataset["test"],
    #     shuffle=True,
    #     collate_fn=data_collator,
    #     batch_size=max_batch_size,
    #     pin_memory=True,
    # )

    print("Setup Model")
    model = LlamaForCausalLM.from_pretrained(
        model_path,
    )
    training_args = TrainingArguments(
        output_dir="output/",
        do_train=True,
        auto_find_batch_size=True,
        gradient_accumulation_steps=10,
        evaluation_strategy="epoch",
        fsdp=["full_shard", "auto_wrap"],
        save_strategy="epoch",
        fp16=True,
        fsdp_config={
            "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
            "fsdp_backward_prefetch": "backward_pre",
            "fsdp_forward_prefetch": True,
        },
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        args=training_args,
    )
    trainer.train()
    trainer.save_state()


if __name__ == "__main__":
    main()
