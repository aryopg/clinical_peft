import os

import evaluate
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
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

from .configs import Configs
from .utils.common_utils import delete_files_in_directory
from .utils.dataset_utils import preprocess_dataset
from .utils.model_utils import load_peft_config


def train(
    configs: Configs,
    wandb_tracker: WandBTracker,
    accelerator: Accelerator,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    outputs_dir: str,
) -> None:
    # Load Model
    peft_config: PeftConfig = load_peft_config(
        configs.model_configs.peft_type,
        configs.model_configs.task_type,
        wandb_tracker.config,
    )

    if configs.model_configs.task_type == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(
            configs.model_configs.model_name_or_path
        )
    elif configs.model_configs.task_type == "seq_cls":
        model = AutoModelForSequenceClassification.from_pretrained(
            configs.model_configs.model_name_or_path
        )
        roc_auc_metric = evaluate.load("roc_auc")
        f1_metric = evaluate.load("f1")

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=configs.model_configs.model_hyperparameters.learning_rate
    )

    # lr scheduler
    training_steps = min(len(train_dataloader), configs.training_configs.steps)
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

    # Train
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

            if configs.model_configs.task_type == "causal_lm":
                train_loss = loss.detach().float()
                train_ppl = torch.exp(train_loss)
            elif configs.model_configs.task_type == "seq_cls":
                roc_auc = roc_auc_metric.compute(predictions, references)
                f1_micro = f1_metric.compute(predictions, references, average="micro")
                f1_macro = f1_metric.compute(predictions, references, average="macro")

        if (
            train_step + 1
        ) >= training_steps or train_step % configs.training_configs.log_steps == 0:
            accelerator.print(
                f"{train_step=}/{training_steps}: {train_ppl.item()=} - {train_loss.item()=}"
            )
            accelerator.log(
                {
                    "train_loss": train_loss,
                    "train_ppl": train_ppl,
                },
                step=train_step,
            )

        if train_step > 0 and (
            (train_step + 1) >= training_steps
            or train_step % configs.training_configs.eval_steps == 0
        ):
            model.eval()
            total_loss = 0
            for eval_step, batch in enumerate(eval_dataloader):
                batch = {k: v for k, v in batch.items() if k not in ["token_type_ids"]}
                with torch.no_grad():
                    outputs = model(**batch)
                    loss = outputs.loss
                    total_loss += loss.detach().float()
            eval_loss = total_loss / len(eval_dataloader)
            eval_ppl = torch.exp(eval_loss)
            accelerator.print(
                f"{train_step=}/{training_steps}: {train_ppl.item()=} - {train_loss.item()=} - {eval_ppl.item()=} - {eval_loss.item()=}"
            )
            accelerator.log(
                {
                    "train_loss": train_loss,
                    "train_ppl": train_ppl,
                    "eval_loss": eval_loss,
                    "eval_ppl": eval_ppl,
                },
                step=train_step,
            )

        if (train_step + 1) >= training_steps:
            break

    accelerator.wait_for_everyone()

    # with accelerator.is_main_process:
    # accelerator.save_state(output_dir=os.path.join(outputs_dir, "checkpoint"))
    # wandb_tracker.save(os.path.join(outputs_dir, "checkpoint"))

    # # Clean up state files to not fill in the memory
    # delete_files_in_directory(os.path.join(outputs_dir, "checkpoint"))

    hf_username = os.getenv("HF_USERNAME")
    hf_upload_token = os.getenv("HF_UPLOAD_TOKEN")
    model_name = configs.model_configs.model_name_or_path.split("/")[-1]
    hyperparams = []
    for key, value in wandb_tracker.config.items():
        hyperparams += [f"{key}_{value}"]
    hyperparams = "__".join(hyperparams)

    hf_repo_name = f"{hf_username}/{model_name}__{peft_config.peft_type}__{hyperparams}"

    huggingface_hub.create_repo(
        hf_repo_name, private=True, token=hf_upload_token, repo_type="model"
    )
    model.push_to_hub(
        hf_repo_name,
        state_dict=accelerator.get_state_dict(model),
        private=True,
        use_auth_token=hf_upload_token,
    )

    accelerator.wait_for_everyone()
    
    # cleanup and sleep just to be sure the cuda memory is freed
    del model
    torch.cuda.empty_cache()
    time.sleep(10)


def run_sweep(
    accelerator: Accelerator,
    configs: Configs,
    wandb_entity: str,
    wandb_project: str,
    outputs_dir: str,
) -> None:
    # Initialise tracker
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=wandb_project, init_kwargs={"wandb": {"entity": wandb_entity}}
        )
        wandb_tracker: WandBTracker = accelerator.get_tracker("wandb")
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
