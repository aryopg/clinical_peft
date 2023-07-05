import ast

import numpy as np
from datasets import DatasetDict
from transformers import PreTrainedTokenizer

from ..configs import Configs, TaskType
from ..constants import IOB_NER_MAP, LABELS_MAP


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


def tokenize(
    dataset: DatasetDict,
    configs: Configs,
    tokenizer: PreTrainedTokenizer,
    text_column: str = "text",
    columns_to_remove: list = [],
):
    processed_datasets = dataset.map(
        lambda x: tokenizer(
            x[text_column],
            max_length=configs.model_configs.model_hyperparameters.max_seq_len,
            truncation=True,
        ),
        batched=True,
        num_proc=configs.training_configs.num_process,
        remove_columns=columns_to_remove,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    return processed_datasets


def preprocess_qa_dataset(
    dataset: DatasetDict, configs: Configs, tokenizer: PreTrainedTokenizer, train: bool
):
    questions = [q.strip() for q in dataset["question"]]
    processed_datasets = tokenizer(
        questions,
        dataset["context"],
        max_length=configs.model_configs.model_hyperparameters.max_seq_len,
        truncation="only_second",
        stride=configs.model_configs.model_hyperparameters.max_seq_len // 2,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = processed_datasets.pop("overflow_to_sample_mapping")

    if train:
        offset_mapping = processed_datasets.pop("offset_mapping")
        answers = dataset["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx][0]
            start_char = answer["answer_start"]
            end_char = answer["answer_start"] + len(answer["text"])
            sequence_ids = processed_datasets.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if (
                offset[context_start][0] > start_char
                or offset[context_end][1] < end_char
            ):
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        processed_datasets["start_positions"] = start_positions
        processed_datasets["end_positions"] = end_positions
    else:
        example_ids = []

        for i in range(len(processed_datasets["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(dataset["id"][sample_idx])

            sequence_ids = processed_datasets.sequence_ids(i)
            offset = processed_datasets["offset_mapping"][i]
            processed_datasets["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        processed_datasets["example_id"] = example_ids

    return processed_datasets


def preprocess_ner_dataset(
    dataset: DatasetDict, configs: Configs, tokenizer: PreTrainedTokenizer
):
    dataset_name = configs.training_configs.dataset_paths[0].split("/")[-1]

    if dataset_name == "n2c2-2010":
        tokenized_inputs = tokenizer(
            dataset["text"],
            max_length=configs.model_configs.model_hyperparameters.max_seq_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        iob_ner_map = IOB_NER_MAP[dataset_name]

        print("dataset['text'][3]: ", dataset["text"][3])
        print("dataset['concepts'][3]: ", dataset["concepts"][3])
        print("dataset['assertions'][3]: ", dataset["assertions"][3])
        print("dataset['relations'][3]: ", dataset["relations"][3])
        mapped_labels = []
        for tags in dataset[f"concepts"]:
            print("tags: ", tags)
            seq_len = len(tokenized_inputs["input_ids"][0])
            iob_tags = ["O"] * seq_len

            for tag in tags:
                print("tag: ", tag)
                print("tag.keys(): ", tag.keys())
                start = tag["start"]
                end = tag["end"]
                tag_id = tag["concept"]

                # Find the subword token indices corresponding to the character offsets
                token_start = None
                token_end = None

                for idx, (offset_start, offset_end) in enumerate(
                    tokenized_inputs["offset_mapping"][0]
                ):
                    if offset_start <= start and offset_end > start:
                        token_start = idx
                    if offset_start < end and offset_end >= end:
                        token_end = idx

                if token_start is not None and token_end is not None:
                    if token_start == token_end:
                        # Entity is within a single subword token
                        iob_tags[token_start] = "B-" + str(tag_id)
                    else:
                        # Entity spans multiple subword tokens
                        iob_tags[token_start] = "B-" + str(tag_id)
                        iob_tags[token_start + 1 : token_end + 1] = [
                            "I-" + str(tag_id)
                        ] * (token_end - token_start)

            label_ids = [iob_ner_map[tag] for tag in iob_tags]
            mapped_labels.append(label_ids)

        tokenized_inputs["labels"] = mapped_labels
    elif dataset_name == "n2c2-2018":
        labels_map = {v: k for k, v in LABELS_MAP[dataset_name].items()}
        iob_ner_map = IOB_NER_MAP[dataset_name]
        tokenized_inputs = tokenizer(
            dataset["text"],
            max_length=configs.model_configs.model_hyperparameters.max_seq_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        mapped_labels = []
        for sample_id, tags in enumerate(dataset[f"tags"]):
            seq_len = len(tokenized_inputs["input_ids"][sample_id])
            iob_tags = ["O"] * seq_len

            offset = tokenized_inputs["offset_mapping"][sample_id]

            for tag in tags:
                start = tag["start"]
                end = tag["end"]
                tag_id = labels_map[tag["tag"]]

                # Find the subword token indices corresponding to the character offsets
                token_start = None
                token_end = None

                for idx, (offset_start, offset_end) in enumerate(offset):
                    if (
                        token_start is None
                        and start >= offset_start
                        and start < offset_end
                    ):
                        token_start = idx
                    if token_end is None and end > offset_start and end <= offset_end:
                        token_end = idx

                if token_start is not None and token_end is not None:
                    if token_start == token_end:
                        # Entity is within a single subword token
                        iob_tags[token_start] = "B-" + str(tag_id)
                    else:
                        # Entity spans multiple subword tokens
                        iob_tags[token_start] = "B-" + str(tag_id)
                        iob_tags[token_start + 1 : token_end + 1] = [
                            "I-" + str(tag_id)
                        ] * (token_end - token_start)

            label_ids = [iob_ner_map[tag] for tag in iob_tags]
            mapped_labels.append(label_ids)

        tokenized_inputs["labels"] = mapped_labels

    return tokenized_inputs


def preprocess_dataset(
    dataset: DatasetDict,
    configs: Configs,
    tokenizer: PreTrainedTokenizer,
) -> DatasetDict:
    if configs.model_configs.task_type == TaskType.causal_lm:
        columns_to_remove = dataset["train"].column_names
        processed_datasets = tokenize(
            dataset, configs, tokenizer, "text", columns_to_remove
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

        columns_to_remove = [
            column_name
            for column_name in dataset["train"].column_names
            if not column_name == "label"
        ]
        processed_datasets = tokenize(
            dataset, configs, tokenizer, "text", columns_to_remove
        )
        processed_datasets = processed_datasets.rename_column("label", "labels")
    elif configs.model_configs.task_type == TaskType.question_ans:
        dataset = dataset.map(
            lambda example: {"answers": ast.literal_eval(example["answers"])},
            remove_columns=["answers"],
        )
        # Remove samples without answers or more than 1 answer
        # TODO: maybe want to explore having a label (NO ANSWER), instead of removing them
        dataset["train"] = dataset["train"].filter(lambda x: len(x["answers"]) == 1)

        processed_datasets = DatasetDict()
        processed_datasets["train"] = dataset["train"].map(
            lambda x: preprocess_qa_dataset(x, configs, tokenizer, train=True),
            batched=True,
            remove_columns=dataset["train"].column_names,
        )
        processed_datasets["validation"] = dataset["validation"].map(
            lambda x: preprocess_qa_dataset(x, configs, tokenizer, train=False),
            batched=True,
            remove_columns=dataset["validation"].column_names,
        )
        processed_datasets["test"] = dataset["test"].map(
            lambda x: preprocess_qa_dataset(x, configs, tokenizer, train=False),
            batched=True,
            remove_columns=dataset["test"].column_names,
        )
    elif configs.model_configs.task_type == TaskType.token_cls:
        processed_datasets = dataset.map(
            lambda x: preprocess_ner_dataset(x, configs, tokenizer),
            batched=True,
            remove_columns=dataset["train"].column_names,
        )

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
