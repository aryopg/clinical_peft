import os

from dotenv import load_dotenv

load_dotenv("env/.env")

import huggingface_hub
from datasets import Dataset, DatasetDict, load_dataset


def upload_to_huggingface_hub(abstracts, dataset_name):
    dataset_dict = {"abstract": abstracts}
    dataset = Dataset.from_dict(dataset_dict)

    # Save the dataset to a file
    dataset.save_to_disk(f"{dataset_name}")

    # Upload the dataset to the Hugging Face Model Hub
    hf_upload_token = os.getenv("HF_UPLOAD_TOKEN")
    try:
        huggingface_hub.create_repo(
            dataset_name, private=False, token=hf_upload_token, repo_type="dataset"
        )
    except:
        pass
    dataset.push_to_hub(dataset_name, token=hf_upload_token)


if __name__ == "__main__":
    email = "aryo.gema@ed.ac.uk"
    num_abstracts = 10000
    dataset_name = "mini_pubmed"

    biomedical_dataset = load_dataset(
        "monology/pile-uncopyrighted",
        split="train",
        streaming=True,
    )

    abstracts = []
    for batch in biomedical_dataset.take(100000):
        if batch["meta"]["pile_set_name"] == "PubMed Abstracts":
            abstracts += [batch["text"]]
        if len(abstracts) == 10000:
            break

    upload_to_huggingface_hub(abstracts, dataset_name)
