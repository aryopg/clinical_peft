import argparse
import functools
import os
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv("env/.env")

import huggingface_hub
from accelerate import Accelerator

import wandb
from clinical_peft.configs import Configs
from clinical_peft.trainer import run_sweep
from clinical_peft.utils import common_utils


def argument_parser():
    parser = argparse.ArgumentParser(description="Clinical PEFT")
    parser.add_argument("--config_filepath", type=str, required=True)
    parser.add_argument("--existing_sweep_id", type=str, default="")
    args = parser.parse_args()
    return args


def main() -> None:
    args = argument_parser()
    configs = Configs(**common_utils.load_yaml(args.config_filepath))

    common_utils.setup_random_seed(configs.training_configs.random_seed)
    outputs_dir = common_utils.setup_experiment_folder(
        os.path.join(os.getcwd(), configs.training_configs.outputs_dir)
    )
    common_utils.save_training_configs(configs, outputs_dir)

    # Login to HF
    huggingface_hub.login(token=os.getenv("HF_DOWNLOAD_TOKEN", ""))

    # Instantiate Accelerator and Login to WandB
    wandb_entity = os.getenv("WANDB_ENTITY", "")
    wandb_project = os.getenv("WANDB_PROJECT_NAME", "")

    # Start sweep
    sweep_configuration = configs.model_configs.peft_hyperparameters
    dataset_name = configs.training_configs.dataset_paths[0].split("/")[-1]
    model_name = configs.model_configs.model_name_or_path.split("/")[-1]
    sweep_name = f"{dataset_name}__{model_name}__{configs.model_configs.peft_type}"
    sweep_configuration["name"] = sweep_name

    if len(args.existing_sweep_id) > 0:
        sweep_id = args.existing_sweep_id
    else:
        sweep_id = wandb.sweep(
            sweep=sweep_configuration,
            entity=wandb_entity,
            project=wandb_project,
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=configs.model_configs.model_hyperparameters.gradient_accumulation_steps,
        log_with="wandb",
    )
    wandb.agent(
        sweep_id,
        function=functools.partial(
            run_sweep,
            accelerator,
            configs,
            wandb_entity,
            wandb_project,
            sweep_name,
        ),
        entity=wandb_entity,
        project=wandb_project,
        count=configs.training_configs.max_sweep_count,
    )


if __name__ == "__main__":
    main()
