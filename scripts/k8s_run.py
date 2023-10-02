from kubejobs import create_jobs_for_experiments

base_command = "git clone 'https://$GIT_TOKEN@github.com/aryopg/clinical_peft.git' --branch longer_sequence && cd clinical_peft && "

# List of commands to run as separate Kubernetes Jobs
commands = [
    "accelerate launch scripts/train.py --config_filepath configs/mimic_pretrain_hpo_configs/mimic_iv/llama_7b_lora.yaml",
    "accelerate launch scripts/train.py --config_filepath configs/mimic_pretrain_hpo_configs/mimic_iv/llama_13b_lora.yaml",
    "accelerate launch scripts/train.py --config_filepath configs/mimic_pretrain_hpo_configs/mimic_iv/llama2_7b_lora.yaml",
    "accelerate launch scripts/train.py --config_filepath configs/mimic_pretrain_hpo_configs/mimic_iv/llama2_13b_lora.yaml",
    "accelerate launch scripts/train.py --config_filepath configs/mimic_pretrain_hpo_configs/mimic_iv/pmc_llama_7b_lora.yaml",
    "accelerate launch scripts/train.py --config_filepath configs/mimic_pretrain_hpo_configs/mimic_iv/pmc_llama_13b_lora.yaml",
    "accelerate launch scripts/train.py --config_filepath configs/mimic_pretrain_hpo_configs/mimic_iv/medllama_13b_lora.yaml",
]

commands = [base_command + command for command in commands]

name = "clinical-llama"
experiment = "dapt-mimic-iv"

# Create and run Kubernetes Jobs for each command in the list
create_jobs_for_experiments(
    commands,
    image="aryopg/clinical-peft:v1",
    gpu_type="nvidia.com/gpu",  # Request GPU resources
    gpu_limit=1,  # Number of GPU resources to allocate
    backoff_limit=4,  # Maximum number of retries before marking job as failed
    name=name,
    experiment=experiment,
    secret_env_vars={
        "GIT_TOKEN": {"secret_name": "aryo-git-token", "key": "aryo-git-token"},
        "WANDB_API_KEY": {
            "secret_name": "aryo-wandb-api-key",
            "key": "aryo-wandb-api-key",
        },
        "WANDB_PROJECT_NAME": {
            "secret_name": "aryo-wandb-project-name",
            "key": "aryo-wandb-project-name",
        },
        "WANDB_ENTITY": {
            "secret_name": "aryo-wandb-entity",
            "key": "aryo-wandb-entity",
        },
        "HF_DOWNLOAD_TOKEN": {
            "secret_name": "aryo-hf-download-token",
            "key": "aryo-hf-download-token",
        },
        "HF_UPLOAD_TOKEN": {
            "secret_name": "aryo-hf-upload-token",
            "key": "aryo-hf-upload-token",
        },
        "HF_USERNAME": {"secret_name": "aryo-hf-username", "key": "aryo-hf-username"},
    },
)
