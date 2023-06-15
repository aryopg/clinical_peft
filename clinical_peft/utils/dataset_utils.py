import numpy as np
from datasets import DatasetDict
from transformers import PreTrainedTokenizer

from ..configs import Configs, TaskType
from ..constants import LABELS_MAP


def preprocess_multilabel(sample, dataset_name):
    labels = np.zeros(
        [
            len(LABELS_MAP[dataset_name]),
        ]
    )
    for label in sample["label"].split(","):
        labels[LABELS_MAP[dataset_name][label]] = 1

    sample["label"] = labels

    return sample


def preprocess_dataset(
    dataset: DatasetDict,
    configs: Configs,
    tokenizer: PreTrainedTokenizer,
) -> DatasetDict:
    if configs.model_configs.task_type == TaskType.causal_lm:
        processed_datasets = dataset.map(
            lambda x: tokenizer(
                x["text"],
                max_length=configs.model_configs.model_hyperparameters.max_seq_len,
                truncation=True,
            ),
            batched=True,
            num_proc=configs.training_configs.num_process,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
    elif configs.model_configs.task_type == TaskType.seq_cls:
        if configs.training_configs.multilabel:
            dataset = dataset.map(
                lambda x: preprocess_multilabel(
                    x, configs.training_configs.dataset_paths[0].split("/")[-1]
                ),
                num_proc=configs.training_configs.num_process,
                desc="Preprocessing multilabel",
            )

        processed_datasets = dataset.map(
            lambda x: tokenizer(
                x["text"],
                max_length=configs.model_configs.model_hyperparameters.max_seq_len,
                truncation=True,
            ),
            batched=True,
            num_proc=configs.training_configs.num_process,
            remove_columns=[
                column_name
                for column_name in dataset["train"].column_names
                if not column_name == "label"
            ],
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
        processed_datasets = processed_datasets.rename_column("label", "labels")

    if (
        "test" not in dataset
        and configs.training_configs.test_size > 0
        and len(processed_datasets.keys()) == 1
    ):
        # By default, DatasetDict only contains "train" key
        processed_datasets = processed_datasets["train"]

        processed_datasets = processed_datasets.train_test_split(
            test_size=configs.training_configs.test_size, shuffle=True
        )

    return processed_datasets
