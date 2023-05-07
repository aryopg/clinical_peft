from peft import PeftConfig

from clinical_peft.configs import ModelConfigs, PEFTTaskType
from clinical_peft.constants import PEFT_CONFIGS, TASK_TYPE


def load_peft_config(
    peft_type: str, task_type: str, peft_hyperparameters
) -> PeftConfig:
    return PEFT_CONFIGS[peft_type](
        task_type=TASK_TYPE[PEFTTaskType[task_type]],
        inference_mode=False,
        **peft_hyperparameters,
    )
