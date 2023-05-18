from peft import (
    AdaLoraConfig,
    AdaptionPromptConfig,
    LoraConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    TaskType,
)

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
}
