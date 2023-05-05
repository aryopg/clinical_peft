from datasets import DatasetDict
from transformers import PreTrainedTokenizer

from clinical_peft.configs import Configs


def preprocess_dataset(
    dataset: DatasetDict,
    configs: Configs,
    tokenizer: PreTrainedTokenizer,
) -> DatasetDict:
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

    if configs.training_configs.test_size > 0 and len(processed_datasets.keys()) == 1:
        # By default, DatasetDict only contains "train" key
        processed_datasets = processed_datasets["train"]

        processed_datasets = processed_datasets.train_test_split(
            test_size=configs.training_configs.test_size, shuffle=True
        )

    return processed_datasets
