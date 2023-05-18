import os
import time
from typing import Dict, List, Optional

import evaluate
import huggingface_hub
import torch
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
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

from .configs import Configs, PEFTTaskType
from .constants import LABELS_MAP
from .utils.common_utils import delete_files_in_directory
from .utils.dataset_utils import preprocess_dataset
from .utils.model_utils import load_peft_config


def train(
    configs: Configs,
    wandb_tracker: WandBTracker,
    accelerator: Accelerator,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    test_dataloader: Optional[DataLoader] = None,
    sweep_name: str = None,
) -> None:
    # Load Model
    peft_config: PeftConfig = load_peft_config(
        configs.model_configs.peft_type,
        configs.model_configs.task_type,
        wandb_tracker.config,
    )

    if configs.model_configs.task_type == PEFTTaskType.causal_lm:
        model = AutoModelForCausalLM.from_pretrained(
            configs.model_configs.model_name_or_path
        )
    elif configs.model_configs.task_type == PEFTTaskType.seq_cls:
        labels_map = LABELS_MAP[
            configs.training_configs.dataset_paths[0].split("/")[-1]
        ]

        model = AutoModelForSequenceClassification.from_pretrained(
            configs.model_configs.model_name_or_path,
            num_labels=len(labels_map),
            label2id=labels_map,
            id2label={v: k for k, v in labels_map.items()},
        )
        classification_metrics = {
            "roc_auc": evaluate.load("roc_auc"),
            "f1_micro": evaluate.load("f1"),
            "f1_macro": evaluate.load("f1"),
        }

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
        num_warmup_steps=0.06
        * (len(train_dataloader) * configs.training_configs.epochs),
        num_training_steps=training_steps,
    )

    (
        model,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        optimizer,
        lr_scheduler,
    ) = accelerator.prepare(
        model,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        optimizer,
        lr_scheduler,
    )

    # Train
    for epoch in range(configs.training_configs.epochs):
        accelerator.print(f" >>> Epoch {epoch + 1} / {configs.training_configs.epochs}")
        for train_step, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            # Manually remove token type ids
            with accelerator.accumulate(model):
                batch = {k: v for k, v in batch.items() if k != "token_type_ids"}
                print(batch)
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                # for name, param in model.named_parameters():
                #     if "classifier" in name and param.requires_grad:
                #         print(name, param.grad)
                optimizer.zero_grad()

                train_loss = loss.detach().float()

            if (
                (train_step + 1) >= training_steps
                or train_step % configs.training_configs.log_steps == 0
            ):
                metrics = {"train_loss": train_loss}
                if configs.model_configs.task_type == "causal_lm":
                    train_ppl = torch.exp(train_loss)
                    metrics["train_ppl"] = train_ppl

                    accelerator.log(
                        metrics,
                        step=train_step,
                    )

            if (train_step + 1) >= training_steps:
                break
        if configs.model_configs.task_type == "seq_cls":
            accelerator.log(
                metrics,
                step=epoch,
            )

            val_metrics = test(
                accelerator,
                model,
                val_dataloader,
                classification_metrics,
                configs.model_configs.task_type,
                split="val",
            )
            metrics_log = " - ".join(
                [
                    f"{metric_name}: {metric_value}"
                    for metric_name, metric_value in val_metrics.items()
                ]
            )
            accelerator.print(
                f"Epoch: {epoch+1}/{configs.training_configs.epochs}: {metrics_log}"
            )
            accelerator.log(
                val_metrics,
                step=epoch,
            )

    # Evaluate on test data
    test_metrics = test(
        accelerator,
        model,
        test_dataloader,
        classification_metrics,
        configs.model_configs.task_type,
        split="test",
    )
    metrics_log = " - ".join(
        [
            f"{metric_name}: {metric_value}"
            for metric_name, metric_value in test_metrics.items()
        ]
    )
    accelerator.print(f"Test: {metrics_log}")
    accelerator.log(test_metrics)

    accelerator.wait_for_everyone()

    hf_username = os.getenv("HF_USERNAME")
    hf_upload_token = os.getenv("HF_UPLOAD_TOKEN")
    hyperparams = []
    for key, value in wandb_tracker.config.items():
        hyperparams += [f"{key}_{value}"]
    hyperparams = "__".join(hyperparams)

    hf_repo_name = f"{hf_username}/{sweep_name}__{hyperparams}"

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


def test(
    accelerator: Accelerator,
    model: PeftModel,
    dataloader: DataLoader,
    metrics: Dict[str, EvaluationModule],
    task: PEFTTaskType,
    split="val",
) -> dict:
    model.eval()
    total_loss = 0
    samples_seen = 0
    for eval_step, batch in enumerate(tqdm(dataloader)):
        batch = {k: v for k, v in batch.items() if k not in ["token_type_ids"]}
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()

        prediction_scores = F.softmax(outputs.logits, dim=1)[:, -1]
        predictions = outputs.logits.argmax(dim=-1)
        predictions, prediction_scores, references = accelerator.gather(
            (predictions, prediction_scores, batch["labels"])
        )
        # If we are in a multiprocess environment, the last batch has duplicates
        if accelerator.num_processes > 1:
            if eval_step == len(dataloader) - 1:
                predictions = predictions[: len(dataloader.dataset) - samples_seen]
                prediction_scores = prediction_scores[
                    : len(dataloader.dataset) - samples_seen
                ]
                references = references[: len(dataloader.dataset) - samples_seen]
            else:
                samples_seen += references.shape[0]

        if task == PEFTTaskType.seq_cls:
            for metric_name, metric in metrics.items():
                if metric_name == "roc_auc":
                    metric.add_batch(
                        prediction_scores=prediction_scores, references=references
                    )
                else:
                    metric.add_batch(predictions=predictions, references=references)

    eval_loss = total_loss / len(dataloader)

    eval_metrics = {
        f"{split}_loss": eval_loss,
    }
    if task == PEFTTaskType.causal_lm:
        eval_ppl = torch.exp(eval_loss)
        eval_metrics[f"{split}_ppl"] = eval_ppl
    elif task == PEFTTaskType.seq_cls:
        for metric_name, metric in metrics.items():
            eval_metrics[f"{split}_{metric_name}"] = metric.compute()

    return eval_metrics


def run_sweep(
    accelerator: Accelerator,
    configs: Configs,
    wandb_entity: str,
    wandb_project: str,
    sweep_name: str,
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
    if configs.training_configs.dataset_paths[0].endswith("mimic-iv"):
        dataset = load_dataset(
            configs.training_configs.dataset_paths[0], data_files="*.gz"
        )
    else:
        dataset = load_dataset(configs.training_configs.dataset_paths[0])

    # Load Tokenizer
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        configs.model_configs.model_name_or_path
    )

    with accelerator.main_process_first():
        dataset = preprocess_dataset(dataset, configs, tokenizer)
    accelerator.wait_for_everyone()

    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if configs.model_configs.task_type == PEFTTaskType.causal_lm:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    elif configs.model_configs.task_type == PEFTTaskType.seq_cls:
        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            max_length=configs.model_configs.model_hyperparameters.max_seq_len,
        )

    train_dataloader = DataLoader(
        dataset["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=configs.training_configs.batch_size,
        pin_memory=True,
    )
    if "validation" in dataset:
        val_dataloader = DataLoader(
            dataset["validation"],
            shuffle=True,
            collate_fn=data_collator,
            batch_size=configs.training_configs.batch_size,
            pin_memory=True,
        )
    if configs.training_configs.test_size > 0 or "test" in dataset:
        test_dataloader = DataLoader(
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
        val_dataloader,
        test_dataloader,
        sweep_name,
    )
