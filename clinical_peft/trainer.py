import os
import time
from collections import Counter
from typing import Dict, List, Optional

import evaluate
import huggingface_hub
import numpy as np
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
    TrainingArguments,
    get_linear_schedule_with_warmup,
)

from .configs import Configs, TaskType
from .constants import LABELS_MAP, LLAMA_SPECIAL_CHARACTER_IDS, LLAMA_SPECIAL_CHARACTERS
from .utils.common_utils import delete_files_in_directory, setup_random_seed
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
    accelerator.print("Loading model:")
    accelerator.print(configs.model_configs.dict())
    accelerator.print(wandb_tracker.config)

    num_epochs = configs.training_configs.epochs
    # Load Model
    if configs.model_configs.task_type == TaskType.causal_lm:
        model = AutoModelForCausalLM.from_pretrained(
            configs.model_configs.model_name_or_path, return_dict=True
        )
        class_weights = None
        performance_metrics = None
        multi_class = None
    elif configs.model_configs.task_type == TaskType.seq_cls:
        labels_map = LABELS_MAP[
            configs.training_configs.dataset_paths[0].split("/")[-1]
        ]

        model = AutoModelForSequenceClassification.from_pretrained(
            configs.model_configs.model_name_or_path,
            num_labels=len(labels_map),
            label2id=labels_map,
            id2label={v: k for k, v in labels_map.items()},
        )

        train_labels = train_dataloader.dataset["labels"]
        num_train_data = len(train_labels)
        # No class weighting for multilabel classification
        class_weights = None
        if not configs.training_configs.multilabel:
            label_counts = Counter(train_labels)
            class_weights = torch.Tensor(
                [
                    num_train_data / (len(labels_map) * label_counts[i])
                    for i in range(len(labels_map))
                ]
            ).to(accelerator.device)

        multi_class = None
        if not configs.training_configs.multilabel:
            f1_metrics = evaluate.load("f1")
            if len(labels_map) > 2:
                roc_auc_metrics = evaluate.load("roc_auc", "multiclass")
                multi_class = "ovo"
            elif len(labels_map) == 2:
                roc_auc_metrics = evaluate.load("roc_auc")
        else:
            f1_metrics = evaluate.load("f1", "multilabel")
            # roc_auc_metrics = evaluate.load("roc_auc", "multilabel")
            roc_auc_metrics = None

        performance_metrics = {
            "roc_auc": roc_auc_metrics,
            "f1_micro": f1_metrics,
            "f1_macro": f1_metrics,
        }

    if configs.model_configs.peft_type:
        peft_config: PeftConfig = load_peft_config(
            configs.model_configs.peft_type,
            configs.model_configs.task_type,
            wandb_tracker.config,
        )

        accelerator.print(peft_config)

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        if "llama" in configs.model_configs.model_name_or_path.lower():
            for name, param in model.named_parameters():
                if name.startswith("classifier") or name.startswith("score"):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    # optimizer
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=configs.model_configs.model_hyperparameters.learning_rate,
    )

    # lr scheduler
    warmup_steps_ratio = configs.model_configs.model_hyperparameters.warmup_steps_ratio
    if configs.model_configs.task_type == TaskType.causal_lm:
        num_training_steps = min(len(train_dataloader), configs.training_configs.steps)
    elif configs.model_configs.task_type == TaskType.seq_cls:
        num_training_steps = len(train_dataloader) * num_epochs
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps_ratio,
        num_training_steps=num_training_steps,
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

    for epoch in range(num_epochs):
        accelerator.print(f" >>> Epoch {epoch + 1} / {num_epochs}")
        model.train()
        for train_step, batch in enumerate(tqdm(train_dataloader)):
            # Manually remove token type ids
            # Labels are removed from the dict to allow custom computation
            with accelerator.accumulate(model):
                labels = batch["labels"]
                batch = {
                    k: v
                    for k, v in batch.items()
                    if k not in ["token_type_ids", "labels"]
                }

                if class_weights is not None:
                    outputs = model(**batch)
                    loss = F.cross_entropy(outputs.logits, labels, weight=class_weights)
                else:
                    outputs = model(**batch, labels=labels)
                    loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            train_loss = loss.detach().float()

            if (
                (train_step + 1) >= num_training_steps
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

                if (train_step + 1) >= num_training_steps:
                    break
        # For classification task, log metrics at the end of an epoch
        if configs.model_configs.task_type == "seq_cls":
            accelerator.log(
                {"train_loss": train_loss},
                step=epoch,
            )

            train_metrics = test(
                accelerator,
                model,
                train_dataloader,
                configs.model_configs.task_type,
                performance_metrics,
                multi_label=configs.training_configs.multilabel,
                multi_class="ovo" if len(labels_map) > 2 else None,
                split="train",
            )

            val_metrics = test(
                accelerator,
                model,
                val_dataloader,
                configs.model_configs.task_type,
                performance_metrics,
                multi_label=configs.training_configs.multilabel,
                multi_class="ovo" if len(labels_map) > 2 else None,
                split="val",
            )
            train_metrics_log = " - ".join(
                [
                    f"{metric_name}: {metric_value}"
                    for metric_name, metric_value in train_metrics.items()
                ]
            )
            val_metrics_log = " - ".join(
                [
                    f"{metric_name}: {metric_value}"
                    for metric_name, metric_value in val_metrics.items()
                ]
            )
            combined_metrics = train_metrics | val_metrics
            metrics_log = train_metrics_log + " - " + val_metrics_log
            accelerator.print(f"Epoch: {epoch+1}/{num_epochs}: {metrics_log}")
            accelerator.log(
                combined_metrics,
                step=epoch,
            )

    # Evaluate on test data
    test_metrics = test(
        accelerator,
        model,
        test_dataloader,
        configs.model_configs.task_type,
        performance_metrics,
        multi_label=configs.training_configs.multilabel,
        multi_class=multi_class,
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
    accelerator.free_memory()
    del model, optimizer, lr_scheduler
    torch.cuda.empty_cache()
    time.sleep(10)


def test(
    accelerator: Accelerator,
    model: PeftModel,
    dataloader: DataLoader,
    task: TaskType,
    metrics: Optional[Dict[str, EvaluationModule]] = None,
    multi_label: Optional[bool] = False,
    multi_class: Optional[str] = None,
    split="val",
) -> dict:
    model.eval()
    total_loss = 0
    for eval_step, batch in enumerate(tqdm(dataloader)):
        batch = {k: v for k, v in batch.items() if k not in ["token_type_ids"]}
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        total_loss += loss.detach().float()

        if task == TaskType.seq_cls:
            prediction_scores = F.softmax(outputs.logits, dim=1)
            if multi_label:
                probs = F.sigmoid(outputs.logits)
                predictions = torch.where(probs >= 0.5, 1.0, 0.0)
            else:
                predictions = outputs.logits.argmax(dim=-1)
            if not multi_class:
                prediction_scores = prediction_scores[:, -1]
            references = batch["labels"]
            predictions, prediction_scores, references = accelerator.gather(
                (predictions, prediction_scores, batch["labels"])
            )

            for metric_name, metric in metrics.items():
                if metric is None:
                    continue
                if metric_name == "roc_auc":
                    metric.add_batch(
                        prediction_scores=prediction_scores, references=references
                    )
                elif metric_name.startswith("f1_"):
                    metric.add_batch(predictions=predictions, references=references)

    eval_loss = total_loss / len(dataloader)

    eval_metrics = {
        f"{split}_loss": eval_loss,
    }
    if task == TaskType.causal_lm:
        eval_ppl = torch.exp(eval_loss)
        eval_metrics[f"{split}_ppl"] = eval_ppl
    elif task == TaskType.seq_cls:
        for metric_name, metric in metrics.items():
            if metric is None:
                continue
            if metric_name == "roc_auc":
                eval_metrics[f"{split}_{metric_name}"] = metric.compute(
                    multi_class=multi_class
                )["roc_auc"]
            elif metric_name == "f1_micro":
                eval_metrics[f"{split}_{metric_name}"] = metric.compute(
                    average="micro"
                )["f1"]
            elif metric_name == "f1_macro":
                eval_metrics[f"{split}_{metric_name}"] = metric.compute(
                    average="macro"
                )["f1"]

    return eval_metrics


def run(
    accelerator: Accelerator,
    configs: Configs,
    wandb_entity: str,
    wandb_project: str,
    sweep_name: str,
    outputs_dir: str,
) -> None:
    setup_random_seed(configs.training_configs.random_seed)

    # Initialise tracker
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=wandb_project,
            init_kwargs={"wandb": {"entity": wandb_entity, "name": sweep_name}},
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
    if any(
        k in configs.model_configs.model_name_or_path for k in ("gpt", "opt", "bloom")
    ):
        padding_side = "left"
    else:
        padding_side = "right"
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        configs.model_configs.model_name_or_path, padding_side=padding_side
    )

    with accelerator.main_process_first():
        dataset = preprocess_dataset(dataset, configs, tokenizer)
    accelerator.wait_for_everyone()

    # TODO: PMC-LLaMA doesn't specify these special characters
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
    val_dataloader, test_dataloader = None, None
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
        # outputs_dir,
    )

    # cleanup and sleep just to be sure the cuda memory is freed
    accelerator.free_memory()
    torch.cuda.empty_cache()
    time.sleep(10)
