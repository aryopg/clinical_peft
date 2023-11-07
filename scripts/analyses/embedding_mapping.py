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
from transformers import AutoModel, AutoTokenizer

import wandb


def argument_parser():
    parser = argparse.ArgumentParser(description="Clinical PEFT")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--pretrained_llama_path", type=str, required=True)
    parser.add_argument("--clinical_llama_lora_path", type=str)
    parser.add_argument("--downstream_llama_lora_path", type=str)
    args = parser.parse_args()
    return args


def get_text_representation(model, inputs, batch_size=4, embedding_pool="last"):
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i : i + batch_size]
            batch_input_ids = torch.stack([torch.tensor(x.ids) for x in batch_inputs])
            batch_attention_mask = torch.stack(
                [torch.tensor(x.attention_mask) for x in batch_inputs]
            )
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
            )
            if embedding_pool == "last":
                batch_embeddings = outputs.last_hidden_state[:, -1, :].numpy()
            embeddings.extend(batch_embeddings)
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
    llama = AutoModel.from_pretrained(args.pretrained_llama_path)
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

    # Load original LLaMA + Clinical LLaMA-LoRA
    clinical_llama_lora = PeftModel.from_pretrained(
        llama,
        args.clinical_llama_lora_path,
    )

    # Get embeddings for each text in the test split
    llama_embeddings = get_text_representation(llama, inputs)
    clinical_llama_lora_embeddings = get_text_representation(
        clinical_llama_lora, inputs
    )

    cols = [f"emb_{i}" for i in range(llama_embeddings.shape[1])]
    llama_embeddings_df = pd.DataFrame(llama_embeddings, columns=cols)
    llama_embeddings_df["LABEL"] = dataset["test"]["label"]

    cols = [f"emb_{i}" for i in range(clinical_llama_lora_embeddings.shape[1])]
    clinical_llama_lora_embeddings_df = pd.DataFrame(
        clinical_llama_lora_embeddings, columns=cols
    )
    clinical_llama_lora_embeddings_df["LABEL"] = dataset["test"]["label"]

    # pca = PCA(n_components=2)
    # llama_embeddings_pca = pca.fit_transform(llama_embeddings)
    # pca = PCA(n_components=2)
    # clinical_llama_lora_embeddings_pca = pca.fit_transform(
    #     clinical_llama_lora_embeddings
    # )

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
