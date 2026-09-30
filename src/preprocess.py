import pandas as pd
import json
import os
import csv


# File locations
DATA_PATH = "data/raw/marketing_campaign.csv"
METADATA_PATH = "data/processed/dataset_metadata.json"


def load_dataset():
    """
    Load the Customer Personality dataset.
    The original CSV file uses tab separation.
    """

    data = pd.read_csv(
        DATA_PATH,
        sep="\t",
        quoting=csv.QUOTE_NONE
    )

    # Clean column names
    data.columns = (
        data.columns
        .astype(str)
        .str.strip()
        .str.strip('"')
        .str.strip("'")
    )

    # Clean Response values
    data["Response"] = (
        data["Response"]
        .astype(str)
        .str.strip()
        .str.strip('"')
        .str.strip("'")
        .astype(int)
    )

    return data


def analyze_dataset(data):
    """
    Display basic information about the dataset.
    """

    print("\n========== DATASET INFORMATION ==========")

    print("\nDataset Shape:")
    print(data.shape)

    print("\nColumn Names:")
    print(data.columns.tolist())

    print("\nData Types:")
    print(data.dtypes)

    print("\nMissing Values:")
    print(data.isnull().sum())

    print("\nDuplicate Rows:")
    print(data.duplicated().sum())

    print("\nTarget Distribution:")
    print(data["Response"].value_counts())

    print("\nCategorical Columns:")
    print(
        data.select_dtypes(
            include=["object"]
        ).columns.tolist()
    )


def save_metadata(data):
    """
    Save basic dataset information
    into dataset_metadata.json.
    """

    metadata = {
        "dataset_name": "Customer Personality Analysis",
        "file_name": "marketing_campaign.csv",
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "target_column": "Response",
        "task": "Binary Classification",
        "dataset_version": "1.0"
    }

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    with open(
        METADATA_PATH,
        "w"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print(
        "\nMetadata saved successfully."
    )

    print(
        "Location:",
        METADATA_PATH
    )


def main():

    print("Loading Customer Personality dataset...")

    data = load_dataset()

    analyze_dataset(data)

    save_metadata(data)

    print("\nPreprocessing and dataset analysis completed.")


if __name__ == "__main__":
    main()