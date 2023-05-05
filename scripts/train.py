import argparse
import os
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv("env/.env")

import huggingface_hub
import wandb
from accelerate import Accelerator

from clinical_peft.configs import Configs
from clinical_peft.trainer import run_sweep
from clinical_peft.utils import common_utils


def argument_parser():
    parser = argparse.ArgumentParser(description="Clinical PEFT")
    parser.add_argument("--config_filepath", type=str, required=True)
    parser.add_argument("--log_to_wandb", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = argument_parser()
    configs = Configs(**common_utils.load_yaml(args.config_filepath))

    common_utils.setup_random_seed(configs.training_configs.random_seed)
    outputs_dir = common_utils.setup_experiment_folder(
        os.path.join(os.getcwd(), configs.training_configs.outputs_dir)
    )
    common_utils.save_training_configs(configs, outputs_dir)

    # Login to HF
    huggingface_hub.login(token=os.getenv("HF_DOWNLOAD_TOKEN", ""))

    sweep_configuration = configs.model_configs.peft_hyperparameters
    print(sweep_configuration)

    wandb_entity = os.getenv("WANDB_ENTITY", "")
    wandb_project = os.getenv("WANDB_PROJECT_NAME", "")
    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        entity=wandb_entity,
        project=wandb_project,
    )

    # Instantiate Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=configs.model_configs.model_hyperparameters.gradient_accumulation_steps,
        log_with="wandb",
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=os.getenv("WANDB_PROJECT_NAME", ""),
            init_kwargs={
                "wandb": {
                    "entity": os.getenv("WANDB_ENTITY", ""),
                    "mode": "online" if args.log_to_wandb else "disabled",
                }
            },
        )
    accelerator.wait_for_everyone()

    wandb.agent(
        sweep_id,
        function=run_sweep(accelerator, configs, outputs_dir),
        count=configs.training_configs.max_sweep_count,
    )


if __name__ == "__main__":
    main()
