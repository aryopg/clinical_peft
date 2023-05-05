import os

import huggingface_hub
import torch
from accelerate import Accelerator
from accelerate.tracking import WandBTracker
from datasets import load_dataset
from peft import PeftConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

import wandb

from .configs import Configs
from .utils.dataset_utils import preprocess_dataset
from .utils.model_utils import load_peft_config


def train(
    configs: Configs,
    wandb_tracker: WandBTracker,
    accelerator: Accelerator,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    outputs_dir: str,
):
    # Load Model
    peft_config: PeftConfig = load_peft_config(
        configs.model_configs.peft_type,
        configs.model_configs.task_type,
        wandb_tracker.config,
    )

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
        num_training_steps=min(len(train_dataloader), configs.training_configs.steps),
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

    # Train
    for train_step, batch in enumerate(tqdm(train_dataloader)):
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
            accelerator.print(
                f"{train_step=}: {train_ppl.item()=} {train_loss.item()=}"
            )

        if (
            train_step > configs.training_configs.steps
            or train_step % configs.training_configs.eval_steps == 0
        ):
            model.eval()
            total_loss = 0
            for batch in tqdm(eval_dataloader):
                batch = {k: v for k, v in batch.items() if k not in ["token_type_ids"]}
                with torch.no_grad():
                    outputs = model(**batch)
                    loss = outputs.loss
                    total_loss += loss.detach().float()
            eval_loss = total_loss / len(eval_dataloader)
            eval_ppl = torch.exp(eval_loss)
            accelerator.print(f"{train_step=}: {eval_ppl.item()=} {eval_loss.item()=}")
            accelerator.log(
                {
                    "train_loss": train_loss,
                    "train_ppl": train_ppl,
                    "eval_loss": eval_loss,
                    "eval_ppl": eval_ppl,
                },
                step=train_step,
            )

        if (
            train_step > configs.training_configs.steps
            or train_step % configs.training_configs.checkpoint_steps == 0
        ):
            with accelerator.is_main_process:
                accelerator.save_state(
                    output_dir=os.path.join(outputs_dir, "checkpoint")
                )
                wandb_tracker.save(os.path.join(outputs_dir, "checkpoint"))
            accelerator.wait_for_everyone()

        if train_step >= configs.training_configs.steps:
            break

    accelerator.wait_for_everyone()


def run_sweep(
    accelerator: Accelerator,
    configs: Configs,
    wandb_entity: str,
    wandb_project: str,
    outputs_dir: str,
):
    # Initialise tracker
    if accelerator.is_main_process:
        accelerator.init_trackers(project_name=wandb_project)
        wandb_tracker: WandBTracker = accelerator.get_tracker("wandb")
        print(wandb_tracker.tracker.config)
    accelerator.wait_for_everyone()

    # Load dataset
    # TODO: Allow multiple datasets load
    dataset = load_dataset(configs.training_configs.dataset_paths[0], data_files="*.gz")

    # Load Tokenizer
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        configs.model_configs.model_name_or_path
    )

    with accelerator.main_process_first():
        dataset = preprocess_dataset(dataset, configs, tokenizer)
    accelerator.wait_for_everyone()

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_dataloader = DataLoader(
        dataset["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=configs.training_configs.batch_size,
        pin_memory=True,
    )
    if configs.training_configs.test_size > 0:
        eval_dataloader = DataLoader(
            dataset["test"],
            shuffle=True,
            collate_fn=data_collator,
            batch_size=configs.training_configs.batch_size,
            pin_memory=True,
        )

    train(
        configs,
        wandb_tracker.tracker,
        accelerator,
        train_dataloader,
        eval_dataloader,
        outputs_dir,
    )
