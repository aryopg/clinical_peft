import datetime
import os
import time
from typing import Dict, List, Optional

import evaluate
import huggingface_hub
import torch
from accelerate import Accelerator
from accelerate.tracking import WandBTracker
from accelerate.utils import find_executable_batch_size
from datasets import DatasetDict, load_dataset
from evaluate import EvaluationModule
from peft import PeftConfig, PeftModel, get_peft_model
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    DefaultDataCollator,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

from .configs import Configs, TaskType
from .constants import (
    IOB_NER_MAP,
    LABELS_MAP,
    LLAMA_SPECIAL_CHARACTER_IDS,
    LLAMA_SPECIAL_CHARACTERS,
    PEFT_CONFIGS,
)
from .models.llama import LlamaForQuestionAnswering, LlamaForTokenClassification
from .utils.common_utils import setup_random_seed
from .utils.dataset_utils import preprocess_dataset
from .utils.model_utils import load_peft_config, set_class_weights, set_metrics


def setup_data_loader(
    configs: Configs,
    tokenizer: PreTrainedTokenizer,
    dataset: DatasetDict,
    batch_size: int = 16,
):
    if configs.model_configs.task_type in [TaskType.causal_lm]:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    elif configs.model_configs.task_type == TaskType.seq_cls:
        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding="longest",
            return_tensors="pt",
        )
    elif configs.model_configs.task_type == TaskType.token_cls:
        data_collator = DataCollatorForTokenClassification(
            tokenizer=tokenizer, padding="longest", return_tensors="pt"
        )
    elif configs.model_configs.task_type == TaskType.question_ans:
        data_collator = DefaultDataCollator(
            return_tensors="pt",
        )

    train_dataloader = DataLoader(
        dataset["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size,
        pin_memory=True,
    )
    val_dataloader, test_dataloader = None, None
    if "validation" in dataset:
        val_dataloader = DataLoader(
            dataset["validation"],
            shuffle=True,
            collate_fn=data_collator,
            batch_size=batch_size,
            pin_memory=True,
        )
    if configs.training_configs.test_size > 0 or "test" in dataset:
        test_dataloader = DataLoader(
            dataset["test"],
            shuffle=True,
            collate_fn=data_collator,
            batch_size=batch_size,
            pin_memory=True,
        )

    return train_dataloader, val_dataloader, test_dataloader


def train(
    configs: Configs,
    wandb_tracker: Optional[WandBTracker],
    accelerator: Accelerator,
    tokenizer: PreTrainedTokenizer,
    dataset: DatasetDict,
    sweep_name: str = None,
) -> None:
    wandb_tracker_config = wandb_tracker.config if wandb_tracker is not None else None

    accelerator.print("Loading model:")
    accelerator.print(configs.model_configs.dict())
    accelerator.print(wandb_tracker_config)

    @find_executable_batch_size(starting_batch_size=configs.training_configs.batch_size)
    def inner_training_loop(max_batch_size):
        # Setup data loader
        accelerator.print(max_batch_size)
        train_dataloader, val_dataloader, test_dataloader = setup_data_loader(
            configs, tokenizer, dataset, max_batch_size
        )

        num_epochs = configs.training_configs.epochs
        # Load Model
        labels_map = None
        use_bf16 = configs.model_configs.model_hyperparameters.bf16
        if configs.model_configs.task_type == TaskType.causal_lm:
            model = AutoModelForCausalLM.from_pretrained(
                configs.model_configs.model_name_or_path,
                torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
                return_dict=True,
            )
        elif configs.model_configs.task_type == TaskType.seq_cls:
            labels_map = LABELS_MAP[
                configs.training_configs.dataset_paths[0].split("/")[-1]
            ]

            model = AutoModelForSequenceClassification.from_pretrained(
                configs.model_configs.model_name_or_path,
                num_labels=len(labels_map),
                label2id=labels_map,
                id2label={v: k for k, v in labels_map.items()},
                torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
            )
        elif configs.model_configs.task_type == TaskType.question_ans:
            if "llama" in configs.model_configs.model_name_or_path:
                model = LlamaForQuestionAnswering.from_pretrained(
                    configs.model_configs.model_name_or_path,
                    return_dict=True,
                    torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
                )
            else:
                model = AutoModelForQuestionAnswering.from_pretrained(
                    configs.model_configs.model_name_or_path,
                    return_dict=True,
                    torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
                )
        elif configs.model_configs.task_type == TaskType.token_cls:
            labels_map = IOB_NER_MAP[
                configs.training_configs.dataset_paths[0].split("/")[-1]
            ]
            if "llama" in configs.model_configs.model_name_or_path:
                model = LlamaForTokenClassification.from_pretrained(
                    configs.model_configs.model_name_or_path,
                    num_labels=len(labels_map),
                    label2id=labels_map,
                    id2label={v: k for k, v in labels_map.items()},
                    return_dict=True,
                    torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
                )
            else:
                model = AutoModelForTokenClassification.from_pretrained(
                    configs.model_configs.model_name_or_path,
                    num_labels=len(labels_map),
                    label2id=labels_map,
                    id2label={v: k for k, v in labels_map.items()},
                    return_dict=True,
                    torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
                )

        class_weights = None
        performance_metrics = None
        multi_class = None
        if configs.model_configs.task_type == TaskType.seq_cls and labels_map:
            # Setup class weighting
            if not configs.training_configs.multilabel:
                class_weights = set_class_weights(
                    train_dataloader.dataset["labels"], labels_map, use_bf16
                ).to(accelerator.device)

            # Setup performance metrics
            performance_metrics, multi_class = set_metrics(
                labels_map, configs.training_configs.multilabel
            )
        elif configs.model_configs.task_type == TaskType.token_cls and labels_map:
            class_weights = set_class_weights(
                [
                    label
                    for sample_labels in dataset["train"]["labels"]
                    for label in sample_labels
                ],
                labels_map,
                use_bf16,
            ).to(accelerator.device)
            performance_metrics = {"seqeval": evaluate.load("seqeval")}

        if configs.model_configs.pretrained_peft_name_or_path:
            # Load the Lora model
            model = PeftModel.from_pretrained(
                model,
                configs.model_configs.pretrained_peft_name_or_path,
                is_trainable=configs.model_configs.pretrained_peft_fine_tune,
            )

            if configs.model_configs.downstream_peft:
                pretrained_peft_type = model.peft_config["default"].peft_type
                downstream_peft_config = PEFT_CONFIGS[pretrained_peft_type.lower()](
                    task_type=model.peft_config["default"].task_type,
                    inference_mode=False,
                    **wandb_tracker_config,
                )

                model.add_adapter("lora_downstream", downstream_peft_config)

            for name, param in model.named_parameters():
                if ".score" in name or ".classifier" in name:
                    param.requires_grad = True

            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(name)

            model.print_trainable_parameters()
        else:
            if configs.model_configs.peft_type:
                peft_config: PeftConfig = load_peft_config(
                    configs.model_configs.peft_type,
                    configs.model_configs.task_type,
                    wandb_tracker_config,
                )

                accelerator.print(peft_config)

                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()
            else:
                if configs.model_configs.task_type == TaskType.seq_cls:
                    if "llama" in configs.model_configs.model_name_or_path.lower():
                        for name, param in model.named_parameters():
                            if name.startswith("classifier") or name.startswith(
                                "score"
                            ):
                                param.requires_grad = True
                            else:
                                param.requires_grad = False

        # optimizer
        optimizer = torch.optim.AdamW(
            params=[param for param in model.parameters() if param.requires_grad],
            lr=configs.model_configs.model_hyperparameters.learning_rate,
        )

        # lr scheduler
        warmup_steps_ratio = (
            configs.model_configs.model_hyperparameters.warmup_steps_ratio
        )
        # if configs.model_configs.task_type == TaskType.causal_lm:
        #     num_training_steps = min(len(train_dataloader), configs.training_configs.steps)
        # elif configs.model_configs.task_type == TaskType.seq_cls:
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

        if accelerator.is_main_process:
            accelerator.print(f"Starting training at: {datetime.datetime.now()}")

        if torch.cuda.device_count() > 1:
            accelerator.sync_gradients = False

        for epoch in range(num_epochs):
            accelerator.print(f" >>> Epoch {epoch + 1} / {num_epochs}")
            model.train()
            for train_step, batch in enumerate(tqdm(train_dataloader)):
                # Manually remove token type ids
                # Labels are removed from the dict to allow custom computation
                with accelerator.accumulate(model):
                    if "labels" in batch:
                        labels = batch["labels"]
                        batch = {
                            k: v
                            for k, v in batch.items()
                            if k not in ["token_type_ids", "offset_mapping", "labels"]
                        }

                        if class_weights is not None:
                            outputs = model(**batch)
                            loss = F.cross_entropy(
                                outputs.logits.view(-1, len(labels_map)),
                                labels.view(-1),
                                weight=class_weights,
                            )
                        else:
                            outputs = model(**batch, labels=labels)
                            loss = outputs.loss
                    else:
                        outputs = model(**batch)
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
            # For classification tasks, log metrics at the end of an epoch
            if configs.model_configs.task_type in ["seq_cls", "token_cls"]:
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
                    multi_class=multi_class,
                    split="train",
                    label_list=list(labels_map.keys()),
                )

                val_metrics = test(
                    accelerator,
                    model,
                    val_dataloader,
                    configs.model_configs.task_type,
                    performance_metrics,
                    multi_label=configs.training_configs.multilabel,
                    multi_class=multi_class,
                    split="val",
                    label_list=list(labels_map.keys()),
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
            label_list=list(labels_map.keys()),
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
        for key, value in wandb_tracker_config.items():
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

    inner_training_loop()


def test(
    accelerator: Accelerator,
    model: PeftModel,
    dataloader: DataLoader,
    task: TaskType,
    metrics: Optional[Dict[str, EvaluationModule]] = None,
    multi_label: Optional[bool] = False,
    multi_class: Optional[str] = None,
    split: str = "val",
    label_list: list = [],
) -> dict:
    model.eval()
    total_loss = 0
    for eval_step, batch in enumerate(tqdm(dataloader)):
        batch = {
            k: v
            for k, v in batch.items()
            if k not in ["token_type_ids", "offset_mapping"]
        }
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
            if not multi_class and not multi_label:
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
        elif task == TaskType.token_cls:
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather((predictions, batch["labels"]))

            true_predictions, true_labels = [], []
            for prediction, label in zip(predictions, references):
                true_prediction, true_label = [], []
                for p, l in zip(prediction, label):
                    if l != -100:
                        true_prediction += [label_list[p]]
                        true_label += [label_list[l]]
                true_predictions += [true_prediction]
                true_labels += [true_label]

            metrics["seqeval"].add_batch(
                predictions=true_predictions, references=true_labels
            )

    eval_loss = total_loss / len(dataloader)

    eval_metrics = {
        f"{split}_loss": eval_loss,
    }
    if task == TaskType.causal_lm:
        eval_ppl = torch.exp(eval_loss)
        eval_metrics[f"{split}_ppl"] = eval_ppl
    elif task in [TaskType.seq_cls, TaskType.token_cls]:
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
            elif metric_name == "seqeval":
                ner_metrics = metric.compute(scheme="IOB2")
                eval_metrics[f"{split}_precision"] = ner_metrics["overall_precision"]
                eval_metrics[f"{split}_recall"] = ner_metrics["overall_recall"]
                eval_metrics[f"{split}_f1"] = ner_metrics["overall_f1"]
                eval_metrics[f"{split}_accuracy"] = ner_metrics["overall_accuracy"]

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
    wandb_tracker = None
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

    if (
        getattr(tokenizer, "pad_token_id") is None
        or getattr(tokenizer, "pad_token") is None
    ):
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    with accelerator.main_process_first():
        dataset = preprocess_dataset(dataset, configs, tokenizer)
    accelerator.wait_for_everyone()

    train(
        configs,
        wandb_tracker.tracker if wandb_tracker is not None else None,
        accelerator,
        tokenizer,
        dataset,
        sweep_name,
        # outputs_dir,
    )

    # cleanup and sleep just to be sure the cuda memory is freed
    accelerator.free_memory()
    torch.cuda.empty_cache()
    time.sleep(10)
