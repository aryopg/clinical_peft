from kubejobs.jobs import KubernetesJob

base_command = "git clone 'https://$GIT_TOKEN@github.com/aryopg/clinical_peft.git' --branch longer_sequence && cd clinical_peft && accelerate launch scripts/train.py --config_filepath "

# List of commands to run as separate Kubernetes Jobs
configs = [
    "configs/mimic_pretrain_hpo_configs/mimic_iv/llama_7b_lora.yaml",
    "configs/mimic_pretrain_hpo_configs/mimic_iv/llama_13b_lora.yaml",
    "configs/mimic_pretrain_hpo_configs/mimic_iv/llama2_7b_lora.yaml",
    "configs/mimic_pretrain_hpo_configs/mimic_iv/llama2_13b_lora.yaml",
    "configs/mimic_pretrain_hpo_configs/mimic_iv/pmc_llama_7b_lora.yaml",
    "configs/mimic_pretrain_hpo_configs/mimic_iv/pmc_llama_13b_lora.yaml",
    "configs/mimic_pretrain_hpo_configs/mimic_iv/medllama_13b_lora.yaml",
]

run_names = [
    "-".join(config.split("/")[-2:]).replace(".yaml", "").replace("_", "-")
    for config in configs
]
commands = [base_command + config for config in configs]

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
    job = KubernetesJob(
        name=run_name,
        image="aryopg/clinical-peft:v1",
        gpu_type="nvidia.com/gpu",
        gpu_limit=1,
        gpu_product="NVIDIA-A100-SXM4-80GB",
        backoff_limit=4,
        command=["/bin/bash", "-c", command],
        secret_env_vars=secret_env_vars,
    )

    # Run the Job on the Kubernetes cluster
    job.run()
