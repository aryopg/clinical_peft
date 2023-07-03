import json
import os

import numpy as np
import pandas as pd

RADQA_DIR = "data/dataset/downstream/radqa"
RADQA_FILEPATHS = {
    "train": "train.json",
    "validation": "dev.json",
    "test": "test.json",
}


def radqa_to_df(filepath, record_path=["data", "paragraphs", "qas", "answers"]):
    file = json.loads(open(filepath).read())
    answers = pd.io.json.json_normalize(file, record_path[:-1])
    qas = pd.io.json.json_normalize(file, record_path[:-2])

    idx = np.repeat(qas["context"].values, qas.qas.str.len())
    answers["context"] = idx
    df = answers[["id", "question", "context", "answers"]].set_index("id").reset_index()
    df["c_id"] = df["context"].factorize()[0]

    return df


def main():
    for data_split, filepath in RADQA_FILEPATHS.items():
        df = radqa_to_df(os.path.join(RADQA_DIR, filepath))
        df.to_csv(os.path.join(RADQA_DIR, data_split + ".csv"), index=False)


if __name__ == "__main":
    main()
