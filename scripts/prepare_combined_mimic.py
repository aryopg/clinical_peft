import argparse
import gzip
import os
import re

import pandas as pd

RAW_NOTES_FILEPATH = {
    "mimic_iii": ["NOTEEVENTS.csv.gz"],
    "mimic_iv": ["discharge.csv.gz", "radiology.csv.gz"],
}


def argument_parser():
    parser = argparse.ArgumentParser(description="Prepare MIMIC-IV dataset")
    parser.add_argument("--mimic_iii_dataset_dir", type=str, required=True)
    parser.add_argument("--mimic_iv_dataset_dir", type=str, required=True)
    parser.add_argument(
        "--output_filepath",
        type=str,
        default="data/dataset/mimic-combined/notes.txt.gz",
    )
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

    if not os.path.isdir(os.path.dirname(args.output_filepath)):
        os.mkdir(os.path.dirname(args.output_filepath))

    # Prepare both discharge summaries and radiology reports
    raw_filepaths = []
    for key, clinical_notes_filepaths in RAW_NOTES_FILEPATH.items():
        for clinical_notes_filepath in clinical_notes_filepaths:
            raw_filepaths += [
                (
                    key,
                    os.path.join(
                        getattr(args, f"{key}_dataset_dir"), clinical_notes_filepath
                    ),
                )
            ]

    # Do a standard preprocessing, removing common noises
    dataset = []
    for mimic_version, filepath in raw_filepaths:
        dataset_df = pd.read_csv(filepath, compression="gzip")
        text_column = "text" if mimic_version == "mimic_iv" else "TEXT"
        cleaned_text = (
            dataset_df[text_column].apply(lambda x: clean_clinical_note(x)).values
        )
        print(cleaned_text[:5])
        dataset += list(cleaned_text)

    # Write the combined clinical notes to the output_filepath
    with gzip.open(args.output_filepath, "wt") as file:
        for clinical_note in dataset:
            file.write(f"{clinical_note}\n")


if __name__ == "__main__":
    main()
