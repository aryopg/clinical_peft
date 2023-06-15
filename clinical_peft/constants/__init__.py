from peft import (
    AdaLoraConfig,
    AdaptionPromptConfig,
    LoraConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    TaskType,
)

from .diagnoses import DIAGNOSES_MAP
from .procedures import PROCEDURES_MAP

TASK_TYPE = {"causal_lm": TaskType.CAUSAL_LM, "seq_cls": TaskType.SEQ_CLS}
PEFT_CONFIGS = {
    "lora": LoraConfig,
    "prefix_tuning": PrefixTuningConfig,
    "p_tuning": PromptEncoderConfig,
    "prompt_tuning": PromptTuningConfig,
    "adalora": AdaLoraConfig,
    "adaptation_prompt_tuning": AdaptionPromptConfig,
}

LABELS_MAP = {
    "trec-mortality": {"NOT_DIE": 0, "DIE": 1},
    "trec-pmv": {"LT1W": 0, "GT1W": 1},
    "trec-los": {"LT3D": 0, "3DTO7D": 1, "1WTO2W": 2, "GT2W": 3},
    "core-diagnosis": DIAGNOSES_MAP,
    "core-procedure": PROCEDURES_MAP,
}

# PMC-LLaMA does not use these special characters, we're reintroducing it
LLAMA_SPECIAL_CHARACTERS = {"unk": "<unk>", "bos": "<s>", "pad": "</s>", "eos": "</s>"}
LLAMA_SPECIAL_CHARACTER_IDS = {"unk": 0, "bos": 1, "pad": 2, "eos": 2}
