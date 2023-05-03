import argparse
import os
import sys

from tqdm import tqdm

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv("env/.env")

import huggingface_hub
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import PeftConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

from clinical_peft.configs import Configs, ModelConfigs, PEFTTaskType
from clinical_peft.constants import PEFT_CONFIGS, TASK_TYPE
from clinical_peft.utils import common_utils


def argument_parser():
    parser = argparse.ArgumentParser(description="Clinical PEFT")
    parser.add_argument("--config_filepath", type=str, required=True)
    parser.add_argument("--log_to_wandb", action="store_true")
    args = parser.parse_args()
    return args


def load_peft_config(model_configs: ModelConfigs) -> PeftConfig:
    return PEFT_CONFIGS[model_configs.peft_type](
        task_type=TASK_TYPE[PEFTTaskType[model_configs.task_type]],
        inference_mode=False,
        **model_configs.peft_hyperparameters,
    )


def preprocess_dataset(dataset, configs, accelerator, tokenizer):
    with accelerator.main_process_first():
        processed_datasets = dataset.map(
            lambda x: tokenizer(
                x["text"],
                max_length=configs.model_configs.model_hyperparameters.max_seq_len,
                truncation=True,
            ),
            batched=True,
            num_proc=4,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
    accelerator.wait_for_everyone()

    return processed_datasets["train"]


def main():
    args = argument_parser()
    configs = Configs(**common_utils.load_yaml(args.config_filepath))

    common_utils.setup_random_seed(configs.training_configs.random_seed)
    outputs_dir = common_utils.setup_experiment_folder(
        os.path.join(os.getcwd(), configs.training_configs.outputs_dir)
    )
    common_utils.save_training_configs(configs, outputs_dir)

    # Login to HF
    huggingface_hub.login(token=os.getenv("HF_DOWNLOAD_TOKEN", ""))

    # Instantiate Accelerator and Login to WandB
    accelerator = Accelerator(log_with="wandb")
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=os.getenv("WANDB_PROJECT_NAME", ""),
            init_kwargs={
                "wandb": {
                    "entity": os.getenv("WANDB_ENTITY", ""),
                    "mode": "online" if args.log_to_wandb else "disabled",
                    "config": configs,
                }
            },
        )
    accelerator.wait_for_everyone()

    # Load dataset
    # TODO: Allow multiple datasets load
    dataset = load_dataset(configs.training_configs.dataset_paths[0], data_files="*.gz")

    # TODO: Instantiate trainer class here

    # Load Tokenizer
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        configs.model_configs.model_name_or_path
    )

    train_dataset = preprocess_dataset(dataset, configs, accelerator, tokenizer)

    # FIXME: Use EOS token for padding for the moment
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=configs.training_configs.batch_size,
        pin_memory=True,
    )

    # Load Model
    peft_config: PeftConfig = load_peft_config(configs.model_configs)

    model = AutoModelForCausalLM.from_pretrained(
        configs.model_configs.model_name_or_path
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=configs.model_configs.model_hyperparameters.learning_rate
    )

    # lr scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * configs.training_configs.epochs),
    )

    (
        model,
        train_dataloader,
        optimizer,
        lr_scheduler,
    ) = accelerator.prepare(
        model,
        train_dataloader,
        optimizer,
        lr_scheduler,
    )

    accelerator.print(model)

    for epoch in range(configs.training_configs.epochs):
        # Train
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            # Manually remove token type ids
            batch = {k: v for k, v in batch.items() if k != "token_type_ids"}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")

    accelerator.wait_for_everyone()
    model_name = configs.model_configs.model_name_or_path.replace("data/model", "")
    model_name = model_name.replace("/", "_")
    model.push_to_hub(
        "aryopg/" + f"{model_name}_{peft_config.peft_type}_{peft_config.task_type}",
        state_dict=accelerator.get_state_dict(model),
        use_auth_token=True,
    )


if __name__ == "__main__":
    main()
