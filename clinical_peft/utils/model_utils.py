from peft import PeftConfig

from ..configs import PEFTTaskType
from ..constants import PEFT_CONFIGS, TASK_TYPE


def load_peft_config(
    peft_type: str,
    task_type: str,
    peft_hyperparameters: dict,
    inference_mode: bool = False,
) -> PeftConfig:
    return PEFT_CONFIGS[peft_type](
        task_type=TASK_TYPE[PEFTTaskType[task_type]],
        inference_mode=inference_mode,
        **peft_hyperparameters,
    )
