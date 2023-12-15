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
import plotly.express as px
import torch
from datasets import DatasetDict, load_dataset
from peft import PeftConfig, PeftModel, get_peft_model
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

import wandb
from clinical_peft.utils import common_utils


def argument_parser():
    parser = argparse.ArgumentParser(description="Clinical PEFT")
    parser.add_argument("--llama_embeddings_path", type=str)
    parser.add_argument("--clinical_llama_lora_embeddings_path", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--dim_reduction_algo", type=str, default="pca")
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--specific_icd_code", type=str, default=None)
    parser.add_argument("--random_seed", type=int, default=1234)
    args = parser.parse_args()
    return args


def dimensionality_reduction(embeddings, algorithm, n_components, random_state):
    if algorithm == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
    elif algorithm == "tsne":
        reducer = TSNE(n_components=n_components, random_state=random_state)
    else:
        raise ValueError(
            "Invalid dimensionality reduction algorithm. Use 'pca' or 'tsne'."
        )

    return reducer.fit_transform(embeddings)


def visualise(
    embeddings,
    labels,
    dim,
    dim_reduction_algo,
    model,
    specific_icd_code=None,
    texts=None,
):
    dimensions = [f"Dim{i}" for i in range(1, dim + 1)]
    # Create DataFrames for Plotly
    df_reduced = pd.DataFrame(data=embeddings, columns=dimensions)
    if specific_icd_code:
        df_reduced["Labels"] = [
            1 if specific_icd_code in label.split(",") else 0 for label in labels
        ]
        df_reduced["Original labels"] = labels
    else:
        df_reduced["Labels"] = labels

    if dim == 2:
        fig_reduced = px.scatter(
            df_reduced,
            x="Dim1",
            y="Dim2",
            # text="Original labels",
            hover_name="Original labels",
            # opacity=0,
            color="Labels",
            title=f"{dim_reduction_algo.upper()} Visualization - {model}",
            labels={"Labels": "Labels"},
        )
    elif dim == 3:
        # Create interactive 3D scatter plots
        fig_reduced = px.scatter_3d(
            df_reduced,
            x="Dim1",
            y="Dim2",
            z="Dim3",
            # text="Original labels",
            hover_name="Original labels",
            # opacity=0,
            color="Labels",
            title=f"{dim_reduction_algo.upper()} Visualization - {model}",
            labels={"Labels": "Labels"},
        )

    # Show the plots
    fig_reduced.show()


def main() -> None:
    args = argument_parser()

    huggingface_hub.login(token=os.getenv("HF_DOWNLOAD_TOKEN", ""))

    common_utils.setup_random_seed(args.random_seed)

    clinical_dataset = load_dataset(args.dataset_path, split="train")
    print(f"Number of Clinical dataset: {len(clinical_dataset)}")

    num_dataset = len(clinical_dataset)

    llama_embeddings = pd.read_csv(args.llama_embeddings_path)
    clinical_llama_lora_embeddings = pd.read_csv(
        args.clinical_llama_lora_embeddings_path
    )

    llama_labels = llama_embeddings["LABEL"].values[:num_dataset]
    llama_embeddings = llama_embeddings.iloc[:, :-1].values[:num_dataset]

    clinical_llama_lora_labels = clinical_llama_lora_embeddings["LABEL"].values[
        :num_dataset
    ]
    clinical_llama_lora_embeddings = clinical_llama_lora_embeddings.iloc[:, :-1].values[
        :num_dataset
    ]

    assert (llama_labels == clinical_llama_lora_labels).all()

    # same_embeddings = 0
    # for llama_embedding, clinical_llama_lora_embedding in zip(
    #     llama_embeddings, clinical_llama_lora_embeddings
    # ):
    #     if np.array_equal(llama_embedding, clinical_llama_lora_embedding):
    #         same_embeddings += 1
    # print(f"{same_embeddings}/{len(llama_embeddings)} embeddings are equal")

    # # Perform PCA on embeddings
    # pca = PCA(n_components=2, random_state=args.random_seed)
    # llama_embeddings_pca = pca.fit_transform(llama_embeddings)
    # pca = PCA(n_components=2, random_state=args.random_seed)
    # clinical_llama_lora_embeddings_pca = pca.fit_transform(
    #     clinical_llama_lora_embeddings
    # )

    # print(llama_embeddings_pca)
    # print(clinical_llama_lora_embeddings_pca)

    # # Create a scatter plot for visualization
    # plt.figure(figsize=(12, 6))
    # plt.subplot(121)
    # plt.scatter(
    #     llama_embeddings_pca[:, 0],
    #     llama_embeddings_pca[:, 1],
    #     c=llama_labels,
    #     cmap="viridis",
    # )
    # plt.title("PCA Visualization - LLaMA")
    # plt.colorbar(label="Labels")
    # plt.xlabel("PC 1")
    # plt.ylabel("PC 2")

    # plt.subplot(122)
    # plt.scatter(
    #     clinical_llama_lora_embeddings_pca[:, 0],
    #     clinical_llama_lora_embeddings_pca[:, 1],
    #     c=clinical_llama_lora_labels,
    #     cmap="viridis",
    # )
    # plt.title("PCA Visualization - LLaMA + Clinical LLaMA-LoRA")
    # plt.colorbar(label="Labels")
    # plt.xlabel("PC 1")
    # plt.ylabel("PC 2")

    # plt.tight_layout()
    # plt.show()

    # Perform dimensionality reduction based on the chosen algorithm
    llama_embeddings_reduced = dimensionality_reduction(
        llama_embeddings, args.dim_reduction_algo, args.dim, args.random_seed
    )
    clinical_llama_lora_embeddings_reduced = dimensionality_reduction(
        clinical_llama_lora_embeddings,
        args.dim_reduction_algo,
        args.dim,
        args.random_seed,
    )

    visualise(
        llama_embeddings_reduced,
        llama_labels,
        args.dim,
        args.dim_reduction_algo,
        "LLaMA",
        specific_icd_code=args.specific_icd_code,
        texts=clinical_dataset["text"],
    )
    visualise(
        clinical_llama_lora_embeddings_reduced,
        clinical_llama_lora_labels,
        args.dim,
        args.dim_reduction_algo,
        "LLaMA + Clinical LLaMA-LoRA",
        specific_icd_code=args.specific_icd_code,
        texts=clinical_dataset["text"],
    )


if __name__ == "__main__":
    main()
