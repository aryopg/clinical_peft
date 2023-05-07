from peft import (
    AdaLoraConfig,
    AdaptionPromptConfig,
    LoraConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    TaskType,
)

TASK_TYPE = {"causal_lm": TaskType.CAUSAL_LM}
PEFT_CONFIGS = {
    "lora": LoraConfig,
    "prefix_tuning": PrefixTuningConfig,
    "p_tuning": PromptEncoderConfig,
    "prompt_tuning": PromptTuningConfig,
    "adalora": AdaLoraConfig,
    "adaptation_prompt_tuning": AdaptionPromptConfig,
}
