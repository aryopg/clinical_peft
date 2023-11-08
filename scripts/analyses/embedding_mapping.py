import argparse
import datetime
import functools
import os
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv("env/.env")

import huggingface_hub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict, load_dataset
from peft import PeftConfig, PeftModel, get_peft_model
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

import wandb


def argument_parser():
    parser = argparse.ArgumentParser(description="Clinical PEFT")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--pretrained_llama_path", type=str, required=True)
    parser.add_argument("--clinical_llama_lora_path", type=str)
    parser.add_argument("--downstream_llama_lora_path", type=str)
    args = parser.parse_args()
    return args


def get_text_representation(
    model, inputs, batch_size=4, embedding_pool="last", device=torch.device("cuda:0")
):
    embeddings = []
    for i, name, param in enumerate(model.named_parameters()):
        print(i, name, param.data.size())
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(inputs["input_ids"]), batch_size)):
            batch_input_ids = torch.tensor(inputs["input_ids"][i : i + batch_size]).to(
                device
            )
            batch_attention_mask = torch.tensor(
                inputs["attention_mask"][i : i + batch_size]
            ).to(device)

            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                output_hidden_states=True,
            )
            print(outputs.hidden_states.size())
            last_hidden_states = outputs.hidden_states[-2]

            # Extract embeddings for the last token
            if embedding_pool == "last":
                batch_embeddings = last_hidden_states[:, -1, :].cpu().numpy()
            embeddings.extend(batch_embeddings)
            break
    return embeddings


def main() -> None:
    args = argument_parser()

    # Login to HF
    huggingface_hub.login(token=os.getenv("HF_DOWNLOAD_TOKEN", ""))

    # Login to WandB
    wandb_entity = os.getenv("WANDB_ENTITY", "")
    wandb_project = "clinical_peft_embedding"

    wandb.init(project=wandb_project, entity=wandb_entity)

    # Load dataset
    dataset = load_dataset(args.dataset_path)

    # Load original LLaMA
    llama = AutoModelForCausalLM.from_pretrained(args.pretrained_llama_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_llama_path, padding_side="right"
    )

    if (
        getattr(tokenizer, "pad_token_id") is None
        or getattr(tokenizer, "pad_token") is None
    ):
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    text_list = dataset["test"]["text"]
    inputs = tokenizer(
        text_list,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )

    print(f"Dataset size: {len(inputs['input_ids'])}")

    # Load original LLaMA + Clinical LLaMA-LoRA
    clinical_llama_lora = PeftModel.from_pretrained(
        llama,
        args.clinical_llama_lora_path,
    )

    # Get embeddings for each text in the test split
    print("Get text representation from LLaMA")
    llama_embeddings = get_text_representation(llama, inputs)
    print("Get text representation from LLaMA + Clinical LLaMA-LoRA")
    clinical_llama_lora_embeddings = get_text_representation(
        clinical_llama_lora, inputs
    )
    print("Prep Dataframe")
    cols = [f"emb_{i}" for i in range(llama_embeddings[0].shape[0])]
    llama_embeddings_df = pd.DataFrame(llama_embeddings, columns=cols)
    llama_embeddings_df["LABEL"] = dataset["test"]["label"][: len(llama_embeddings)]

    cols = [f"emb_{i}" for i in range(clinical_llama_lora_embeddings[0].shape[0])]
    clinical_llama_lora_embeddings_df = pd.DataFrame(
        clinical_llama_lora_embeddings, columns=cols
    )
    clinical_llama_lora_embeddings_df["LABEL"] = dataset["test"]["label"][
        : len(clinical_llama_lora_embeddings)
    ]

    # Create a new WandB artifact
    artifact = wandb.Artifact("llama_embeddings_df", type="dataset")
    llama_embeddings_df.to_csv("llama_embeddings_df.csv", index=False)
    artifact.add_file("llama_embeddings_df.csv")
    wandb.log_artifact(artifact)

    artifact = wandb.Artifact("clinical_llama_lora_embeddings_df", type="dataset")
    clinical_llama_lora_embeddings_df.to_csv(
        "clinical_llama_lora_embeddings_df.csv", index=False
    )
    artifact.add_file("clinical_llama_lora_embeddings_df.csv")
    wandb.log_artifact(artifact)

    print("Log to WandB")
    # log pandas DataFrame to W&B easily
    llama_embeddings_table = wandb.Table(
        columns=llama_embeddings_df.columns.to_list(), data=llama_embeddings_df.values
    )
    clinical_llama_lora_embeddings_table = wandb.Table(
        columns=clinical_llama_lora_embeddings_df.columns.to_list(),
        data=clinical_llama_lora_embeddings_df.values,
    )

    wandb.log(
        {
            "LLaMA Embedding": llama_embeddings_table,
            "LLaMA + Clinical LLaMA-LoRA Embedding": clinical_llama_lora_embeddings_table,
        }
    )
    wandb.finish()


if __name__ == "__main__":
    main()
