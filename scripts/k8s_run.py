from kubejobs import create_jobs_for_experiments

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
)
