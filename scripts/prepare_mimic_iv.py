import argparse
import gzip
import os
import re

import pandas as pd

RAW_MIMIC_IV_NOTES_FILEPATHS = ["discharge.csv.gz", "radiology.csv.gz"]


def argument_parser():
    parser = argparse.ArgumentParser(description="Prepare MIMIC-IV dataset")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_filename", type=str, default="notes_cleaned.txt.gz")
    args = parser.parse_args()
    return args


def clean_clinical_note(clinical_note: str) -> str:
    clinical_note = clinical_note.replace("[**", " ")
    clinical_note = clinical_note.replace("**]", " ")
    clinical_note = clinical_note.replace("\n", " ")
    clinical_note = re.sub(" +", " ", clinical_note)

    return clinical_note


def main() -> None:
    args = argument_parser()

    # Prepare both discharge summaries and radiology reports
    raw_filepaths = [
        os.path.join(args.dataset_dir, clinical_notes_filepath)
        for clinical_notes_filepath in RAW_MIMIC_IV_NOTES_FILEPATHS
    ]
    output_filepath = os.path.join(args.dataset_dir, args.output_filename)

    # Do a standard preprocessing, removing common noises
    dataset = []
    for filepath in raw_filepaths:
        dataset_df = pd.read_csv(filepath, compression="gzip")
        dataset += dataset_df["text"].apply(lambda x: clean_clinical_note(x)).values

    # Write the combined clinical notes to the output_filepath
    with gzip.open(output_filepath, "wt") as file:
        for clinical_note in dataset:
            file.write(f"{clinical_note}\n")


if __name__ == "__main__":
    main()
