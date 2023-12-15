from collections import Counter

import evaluate
import torch
from peft import PeftConfig

from ..configs import Configs, TaskType
from ..constants import PEFT_CONFIGS, TASK_TYPE


def load_peft_config(
    peft_type: str,
    task_type: str,
    peft_hyperparameters: dict,
    inference_mode: bool = False,
) -> PeftConfig:
    return PEFT_CONFIGS[peft_type](
        task_type=TASK_TYPE[TaskType[task_type]],
        inference_mode=inference_mode,
        **peft_hyperparameters,
    )


def set_class_weights(train_labels: list, labels_map: dict, bf16: bool = False):
    num_train_data = len(train_labels)

    label_counts = Counter(train_labels)
    class_weights = torch.Tensor(
        [
            num_train_data / (len(labels_map) * label_counts[i])
            for i in range(len(labels_map))
        ]
    )

    if bf16:
        class_weights = class_weights.to(torch.bfloat16)

    return class_weights


def set_metrics(labels_map: dict, multilabel: bool):
    multi_class = None
    if not multilabel:
        f1_micro_metrics = evaluate.load("f1")
        f1_macro_metrics = evaluate.load("f1")
        if len(labels_map) > 2:
            auprc_metrics = evaluate.load("clinical_peft/metrics/auprc", "multiclass")
            roc_auc_metrics = evaluate.load("roc_auc", "multiclass")
            multi_class = "ovo"
        elif len(labels_map) == 2:
            auprc_metrics = evaluate.load("clinical_peft/metrics/auprc")
            roc_auc_metrics = evaluate.load("roc_auc")
    else:
        f1_micro_metrics = evaluate.load("f1", "multilabel")
        f1_macro_metrics = evaluate.load(
            "clinical_peft/metrics/f1_skip_uniform", "multilabel"
        )
        roc_auc_metrics = evaluate.load(
            "clinical_peft/metrics/roc_auc_skip_uniform", "multilabel"
        )
        auprc_metrics = evaluate.load(
            "clinical_peft/metrics/auprc_skip_uniform", "multilabel"
        )

    performance_metrics = {
        "auprc": auprc_metrics,
        "roc_auc": roc_auc_metrics,
        "f1_micro": f1_micro_metrics,
        "f1_macro": f1_macro_metrics,
    }

    return performance_metrics, multi_class
