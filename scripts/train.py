import argparse
import datetime
import functools
import os
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv("env/.env")

import huggingface_hub
import torch.distributed as dist
from accelerate import Accelerator

import wandb
from clinical_peft.configs import Configs
from clinical_peft.trainer import run
from clinical_peft.utils import common_utils


def argument_parser():
    parser = argparse.ArgumentParser(description="Clinical PEFT")
    parser.add_argument("--config_filepath", type=str, required=True)
    parser.add_argument("--existing_sweep_id", type=str, default="")
    args = parser.parse_args()
    return args


def main() -> None:
    try:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(seconds=100000000),
        )
    except ValueError:
        pass

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

    dataset_name = configs.training_configs.dataset_paths[0].split("/")[-1]
    model_name = configs.model_configs.model_name_or_path.split("/")[-1]
    # BlueBERT specific quirk, the name is too long to store on HF hub
    if model_name.startswith("bluebert"):
        model_name = "bluebert"
    if configs.model_configs.pretrained_peft_name_or_path:
        model_name = "clinical_" + model_name
        if configs.model_configs.pretrained_peft_fine_tune is True:
            model_name += "_finetune"
        elif configs.model_configs.pretrained_peft_fine_tune is False:
            model_name += "_frozen"

    if configs.model_configs.peft_hyperparameters:
        # Start sweep
        sweep_configuration = configs.model_configs.peft_hyperparameters
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
                run,
                accelerator,
                configs,
                wandb_entity,
                wandb_project,
                sweep_name,
                outputs_dir,
            ),
            entity=wandb_entity,
            project=wandb_project,
            count=configs.training_configs.max_sweep_count,
        )
    else:
        run_name = f"{dataset_name}__{model_name}__baseline"

        accelerator = Accelerator(
            gradient_accumulation_steps=configs.model_configs.model_hyperparameters.gradient_accumulation_steps,
            log_with="wandb",
        )

        run(accelerator, configs, wandb_entity, wandb_project, run_name, outputs_dir)


if __name__ == "__main__":
    main()
