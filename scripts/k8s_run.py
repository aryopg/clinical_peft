import argparse
import os
import sys

sys.path.append(os.getcwd())

import yaml
from kubejobs.jobs import KubernetesJob


def argument_parser():
    parser = argparse.ArgumentParser(description="Clinical PEFT")
    parser.add_argument("--run_configs_filepath", type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = argument_parser()
    configs = yaml.safe_load(open(args.run_configs_filepath, "r"))

    base_args = "git clone https://$GIT_TOKEN@github.com/aryopg/clinical_peft.git --branch longer_sequence && cd clinical_peft && "
    run_names = [
        "-".join(config["config_filepath"].split("/")[-2:])
        .replace(".yaml", "")
        .replace("_", "-")
        for config in configs["configs"]
    ]
    if configs["gpu_limit"] > 1:
        accelerate_config = configs["accelerate_config"]
        base_command = f"CUDA_LAUNCH_BLOCKING=1 accelerate launch --config_file {accelerate_config} scripts/train.py --config_filepath "
    else:
        base_command = "CUDA_LAUNCH_BLOCKING=1 accelerate launch --mixed_precision bf16 scripts/train.py --config_filepath "

    commands = []
    for config in configs["configs"]:
        command = base_command + config["config_filepath"]
        if config["existing_sweep_id"]:
            existing_sweep_id = config["existing_sweep_id"]
            command += f" --existing_sweep_id {existing_sweep_id}"
        commands += [command]

    secret_env_vars = {
        "GIT_TOKEN": {"secret_name": "aryo-secrets", "key": "aryo-git-token"},
        "WANDB_API_KEY": {
            "secret_name": "aryo-secrets",
            "key": "aryo-wandb-api-key",
        },
        "WANDB_PROJECT_NAME": {
            "secret_name": "aryo-secrets",
            "key": "aryo-wandb-project-name",
        },
        "WANDB_ENTITY": {
            "secret_name": "aryo-secrets",
            "key": "aryo-wandb-entity",
        },
        "HF_DOWNLOAD_TOKEN": {
            "secret_name": "aryo-secrets",
            "key": "aryo-hf-download-token",
        },
        "HF_UPLOAD_TOKEN": {
            "secret_name": "aryo-secrets",
            "key": "aryo-hf-upload-token",
        },
        "HF_USERNAME": {"secret_name": "aryo-secrets", "key": "aryo-hf-username"},
    }

    for run_name, command in zip(run_names, commands):
        # Create a Kubernetes Job with a name, container image, and command
        print(f"Creating job for: {base_args + command}")
        job = KubernetesJob(
            name=run_name,
            image="aryopg/clinical-peft:latest",
            cpu_request="8",
            ram_request="16Gi",
            gpu_type="nvidia.com/gpu",
            gpu_limit=configs["gpu_limit"],
            gpu_product="NVIDIA-A100-SXM4-80GB",
            backoff_limit=4,
            command=["/bin/bash", "-c", "--"],
            args=[base_args + command],
            secret_env_vars=secret_env_vars,
        )

        # Run the Job on the Kubernetes cluster
        job.run()


if __name__ == "__main__":
    main()
