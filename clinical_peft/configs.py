from enum import Enum
from typing import List

from pydantic import BaseModel


class PEFTTaskType(str, Enum):
    causal_lm = "causal_lm"
    seq2seq_lm = "seq2seq_lm"


class PEFTType(str, Enum):
    lora = "lora"
    prefix_tuning = "prefix_tuning"
    p_tuning = "p_tuning"
    prompt_tuning = "prompt_tuning"
    adalora = "adalora"


class ModelHyperparameters(BaseModel):
    learning_rate: float
    max_seq_len: int


class ModelConfigs(BaseModel):
    peft_type: PEFTType
    task_type: PEFTTaskType
    model_name_or_path: str = "data/llama/7B"
    peft_hyperparameters: dict
    model_hyperparameters: ModelHyperparameters

    class Config:
        use_enum_values = True


class TrainingConfigs(BaseModel):
    dataset_paths: List[str]
    random_seed: int = 1234
    device: int = 0
    epochs: int = 1000
    batch_size: int = 256
    grad_accumulation_step: int = 1
    eval_steps: int = 1000
    checkpoint_steps: int = 1000
    outputs_dir: str = "outputs"


class Configs(BaseModel):
    model_configs: ModelConfigs
    training_configs: TrainingConfigs