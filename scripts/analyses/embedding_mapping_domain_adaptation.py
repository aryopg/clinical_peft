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
    parser.add_argument("--random_seed", type=int, default=1234)
    args = parser.parse_args()
    return args


def get_text_representation(
    model, inputs, batch_size=4, embedding_pool="last", device=torch.device("cuda:0")
):
    embeddings = []
    for i, (name, param) in enumerate(model.named_parameters()):
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
            last_hidden_states = outputs.hidden_states[-1]

            # Extract embeddings for the last token
            if embedding_pool == "last":
                batch_embeddings = last_hidden_states[:, -1, :].cpu().numpy()
            embeddings.extend(batch_embeddings)

    return embeddings


def main() -> None:
    args = argument_parser()

    # Login to HF
    huggingface_hub.login(token=os.getenv("HF_DOWNLOAD_TOKEN", ""))

    # Login to WandB
    wandb_entity = os.getenv("WANDB_ENTITY", "")
    wandb_project = "clinical_peft_embedding"

    wandb.init(project=wandb_project, entity=wandb_entity, name=args.dataset_path)

    general_dataset = load_dataset("bookcorpus", split="train", streaming=True)
    general_dataset = [batch["text"] for batch in general_dataset.take(10000)]
    biomedical_dataset = load_dataset("aryopg/mini_pubmed", split="train")["abstract"][
        :10000
    ]
    # Load dataset
    clinical_dataset = load_dataset(args.dataset_path, split="train")["text"][:10000]

    print(f"Number of General dataset: {len(general_dataset)}")
    print(f"Number of Biomedical dataset: {len(biomedical_dataset)}")
    print(f"Number of Clinical dataset: {len(clinical_dataset)}")

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

    text_list = general_dataset + biomedical_dataset + clinical_dataset
    labels_list = (
        [0] * len(general_dataset)
        + [1] * len(biomedical_dataset)
        + [2] * len(clinical_dataset)
    )
    inputs = tokenizer(
        text_list,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )

    print(f"Dataset size: {len(inputs['input_ids'])}")

    # Get embeddings for each text in the test split
    print("Get text representation from LLaMA")
    llama_embeddings = get_text_representation(llama, inputs)

    print("Prep Dataframe from LLaMA")
    cols = [f"emb_{i}" for i in range(llama_embeddings[0].shape[0])]
    llama_embeddings_df = pd.DataFrame(llama_embeddings, columns=cols)
    llama_embeddings_df["LABEL"] = labels_list[: len(llama_embeddings)]

    # Create a new WandB artifact
    print("Create a WandB artifact from LLaMA embedding")
    artifact = wandb.Artifact("llama_embeddings_df", type="dataset")
    llama_embeddings_df.to_csv("llama_embeddings_df.csv", index=False)
    artifact.add_file("llama_embeddings_df.csv")
    wandb.log_artifact(artifact)

    # Load original LLaMA + Clinical LLaMA-LoRA
    clinical_llama_lora = PeftModel.from_pretrained(
        llama,
        args.clinical_llama_lora_path,
    )

    # Debugging LoRA params: lora_B seems to be all zero
    lora_params = {
        n: p for n, p in clinical_llama_lora.named_parameters() if "lora_B" in n
    }
    for n, p in lora_params.items():
        print(n, p.sum())

    print("Get text representation from LLaMA + Clinical LLaMA-LoRA")
    clinical_llama_lora_embeddings = get_text_representation(
        clinical_llama_lora, inputs
    )

    same_embeddings = 0
    for llama_embedding, clinical_llama_lora_embedding in zip(
        llama_embeddings, clinical_llama_lora_embeddings
    ):
        if np.array_equal(llama_embedding, clinical_llama_lora_embedding):
            same_embeddings += 1
    print(f"{same_embeddings}/{len(llama_embeddings)} embeddings are equal")

    print("Prep Dataframe from LLaMA + Clinical LLaMA-LoRA")
    cols = [f"emb_{i}" for i in range(clinical_llama_lora_embeddings[0].shape[0])]
    clinical_llama_lora_embeddings_df = pd.DataFrame(
        clinical_llama_lora_embeddings, columns=cols
    )
    clinical_llama_lora_embeddings_df["LABEL"] = labels_list[
        : len(clinical_llama_lora_embeddings)
    ]

    print("Create a WandB artifact from LLaMA + Clinical LLaMA-LoRA embedding")
    artifact = wandb.Artifact("clinical_llama_lora_embeddings_df", type="dataset")
    clinical_llama_lora_embeddings_df.to_csv(
        "clinical_llama_lora_embeddings_df.csv", index=False
    )
    artifact.add_file("clinical_llama_lora_embeddings_df.csv")
    wandb.log_artifact(artifact)

    print("Perform PCA on embeddings")
    pca = PCA(n_components=2, random_state=args.random_seed)
    llama_embeddings_pca = pca.fit_transform(llama_embeddings)
    pca = PCA(n_components=2, random_state=args.random_seed)
    clinical_llama_lora_embeddings_pca = pca.fit_transform(
        clinical_llama_lora_embeddings
    )

    # Create a scatter plot for visualization
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.scatter(
        llama_embeddings_pca[:, 0],
        llama_embeddings_pca[:, 1],
        c=labels_list,
        cmap="viridis",
    )
    plt.title("PCA Visualization - LLaMA")
    plt.colorbar(label="Labels")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")

    plt.subplot(122)
    plt.scatter(
        clinical_llama_lora_embeddings_pca[:, 0],
        clinical_llama_lora_embeddings_pca[:, 1],
        c=labels_list,
        cmap="viridis",
    )
    plt.title("PCA Visualization - LLaMA + Clinical LLaMA-LoRA")
    plt.colorbar(label="Labels")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")

    # Log the figures to wandb
    wandb.log({"PCA Visualization - LLaMA": plt.figure(1)})
    wandb.log({"PCA Visualization - LLaMA + Clinical LLaMA-LoRA": plt.figure(2)})

    # print("Log to WandB")
    # # log pandas DataFrame to W&B easily
    # llama_embeddings_table = wandb.Table(
    #     columns=llama_embeddings_df.columns.to_list(), data=llama_embeddings_df.values
    # )
    # clinical_llama_lora_embeddings_table = wandb.Table(
    #     columns=clinical_llama_lora_embeddings_df.columns.to_list(),
    #     data=clinical_llama_lora_embeddings_df.values,
    # )

    # wandb.log(
    #     {
    #         "LLaMA Embedding": llama_embeddings_table,
    #         "LLaMA + Clinical LLaMA-LoRA Embedding": clinical_llama_lora_embeddings_table,
    #     }
    # )
    wandb.finish()


if __name__ == "__main__":
    main()
