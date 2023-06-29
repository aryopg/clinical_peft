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

TASK_TYPE = {
    "causal_lm": TaskType.CAUSAL_LM,
    "seq_cls": TaskType.SEQ_CLS,
    "question_ans": TaskType.QUESTION_ANS,
    "token_cls": TaskType.TOKEN_CLS,
}
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
    "n2c2-2018": {
        "ADE": 0,
        "Dosage": 1,
        "Drug": 2,
        "Duration": 3,
        "Form": 4,
        "Frequency": 5,
        "Reason": 6,
        "Route": 7,
        "Strength": 8,
    },
}

IOB_NER_MAP = {
    "n2c2-2018": {
        tag: i
        for i, tag in enumerate(
            [
                "O",
                "B-ADE",
                "I-ADE",
                "B-Dosage",
                "I-Dosage",
                "B-Drug",
                "I-Drug",
                "B-Duration",
                "I-Duration",
                "B-Form",
                "I-Form",
                "B-Frequency",
                "I-Frequency",
                "B-Reason",
                "I-Reason",
                "B-Route",
                "I-Route",
                "B-Strength",
                "I-Strength",
            ]
        )
    }
}

# PMC-LLaMA does not use these special characters, we're reintroducing it
LLAMA_SPECIAL_CHARACTERS = {"unk": "<unk>", "bos": "<s>", "pad": "</s>", "eos": "</s>"}
LLAMA_SPECIAL_CHARACTER_IDS = {"unk": 0, "bos": 1, "pad": 2, "eos": 2}
