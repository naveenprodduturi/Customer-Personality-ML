import pandas as pd
import joblib
import os
import csv
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# File paths
DATA_PATH = "data/raw/marketing_campaign.csv"
MODEL_PATH = "models/logistic_regression_model.pkl"
RESULTS_PATH = "outputs/evaluation_results.json"


def load_dataset():

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


def prepare_data(data):

    X = data.drop(
        columns=["ID", "Response"]
    )

    y = data["Response"]

    # Convert customer date into numerical features
    X["Dt_Customer"] = pd.to_datetime(
        X["Dt_Customer"],
        dayfirst=True,
        errors="coerce"
    )

    X["Customer_Year"] = X["Dt_Customer"].dt.year
    X["Customer_Month"] = X["Dt_Customer"].dt.month
    X["Customer_Day"] = X["Dt_Customer"].dt.day

    X = X.drop(
        columns=["Dt_Customer"]
    )

    return X, y


def main():

    print("Loading dataset...")

    data = load_dataset()

    X, y = prepare_data(data)

    print("Dataset shape:", data.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Testing samples:", X_test.shape[0])

    if not os.path.exists(MODEL_PATH):

        print("\nERROR: Trained model was not found.")

        print(
            "Expected model:",
            MODEL_PATH
        )

        return

    print("\nLoading trained model...")

    model = joblib.load(
        MODEL_PATH
    )

    print("Model loaded successfully.")

    print("\nMaking predictions...")

    y_pred = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        y_pred
    )

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    print("\n========== MODEL EVALUATION ==========")

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print("\nConfusion Matrix:")

    print(matrix)

    print("\nClassification Report:")

    print(report)

    # Create outputs directory
    os.makedirs(
        "outputs",
        exist_ok=True
    )

    # Save results
    results = {
        "model": "Logistic Regression",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": matrix.tolist()
    }

    with open(
        RESULTS_PATH,
        "w"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print(
        "\nEvaluation results saved successfully."
    )

    print(
        "Location:",
        RESULTS_PATH
    )


if __name__ == "__main__":
    main()