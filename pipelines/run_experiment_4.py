import os
import json
import joblib
import mlflow
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ========================================
# EXPERIMENT 4 - MLFLOW EXPERIMENT TRACKING
# ========================================

DATA_PATH = "data/raw/marketing_campaign.csv"
OUTPUT_DIR = "outputs"


def load_dataset():

    print("Loading dataset...")

    data = pd.read_csv(
        DATA_PATH,
        sep="\t",
        engine="python"
    )

    # Fix one-column TSV problem
    if len(data.columns) == 1:

        print("Detected single-column dataset.")
        print("Fixing tab-separated format...")

        with open(
            DATA_PATH,
            "r",
            encoding="utf-8-sig"
        ) as file:

            lines = file.readlines()

        lines = [
            line.rstrip("\n\r")
            for line in lines
            if line.strip()
        ]

        headers = lines[0].split("\t")

        records = []

        for line in lines[1:]:

            values = line.split("\t")

            if len(values) == len(headers):
                records.append(values)

        data = pd.DataFrame(
            records,
            columns=headers
        )

    # Clean column names
    data.columns = (
        data.columns
        .astype(str)
        .str.strip()
        .str.replace('"', '', regex=False)
    )

    # Clean values
    for column in data.columns:

        data[column] = (
            data[column]
            .astype(str)
            .str.strip()
            .str.replace('"', '', regex=False)
        )

    # Numeric columns
    numeric_columns = [
        "Year_Birth",
        "Income",
        "Kidhome",
        "Teenhome",
        "Recency",
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
        "NumDealsPurchases",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
        "NumWebVisitsMonth",
        "AcceptedCmp3",
        "AcceptedCmp4",
        "AcceptedCmp5",
        "AcceptedCmp1",
        "AcceptedCmp2",
        "Complain",
        "Z_CostContact",
        "Z_Revenue"
    ]

    for column in numeric_columns:

        if column in data.columns:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

    # Clean Response
    data["Response"] = (
        data["Response"]
        .astype(str)
        .str.strip()
        .str.replace('"', '', regex=False)
    )

    data["Response"] = pd.to_numeric(
        data["Response"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["Response"]
    )

    data["Response"] = data["Response"].astype(int)

    print("Dataset shape:", data.shape)

    print("\nResponse distribution:")
    print(data["Response"].value_counts())

    print("\nColumn names:")
    print(data.columns.tolist())

    return data


def prepare_data(data):

    target = "Response"

    if target not in data.columns:

        raise ValueError(
            "Response column not found."
        )

    # Remove ID and date
    columns_to_drop = [
        "ID",
        "Dt_Customer"
    ]

    data = data.drop(
        columns=[
            column
            for column in columns_to_drop
            if column in data.columns
        ]
    )

    # Features
    X = data.drop(
        columns=[target]
    )

    # Target
    y = data[target].astype(int)

    # Categorical columns
    categorical_columns = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    # Numerical columns
    numerical_columns = X.select_dtypes(
        exclude=["object", "string"]
    ).columns.tolist()

    print("\nCategorical columns:")
    print(categorical_columns)

    print("\nNumerical columns:")
    print(numerical_columns)

    # Numerical preprocessing
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # Categorical preprocessing
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # Combined preprocessing
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_columns
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_columns
            )
        ]
    )

    return X, y, preprocessing


def run_experiment():

    print("\n" + "=" * 50)
    print("EXPERIMENT 4 - MLFLOW EXPERIMENT TRACKING")
    print("=" * 50)

    # Load dataset
    data = load_dataset()

    # Prepare data
    X, y, preprocessing = prepare_data(data)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # Create MLflow experiment
    mlflow.set_experiment(
        "Customer_Personality_Experiment_4"
    )

    # Model pipeline
    model = Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessing
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )
        ]
    )

    # Start MLflow run
    with mlflow.start_run(
        run_name="Logistic_Regression_Baseline"
    ):

        print("\nTraining Logistic Regression...")

        # Train
        model.fit(
            X_train,
            y_train
        )

        # Predict
        predictions = model.predict(
            X_test
        )

        # Metrics
        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        print("\n----------------------------------------")
        print("MLFLOW EXPERIMENT RESULTS")
        print("----------------------------------------")

        print(
            "Accuracy  :",
            round(accuracy, 4)
        )

        print(
            "Precision :",
            round(precision, 4)
        )

        print(
            "Recall    :",
            round(recall, 4)
        )

        print(
            "F1 Score  :",
            round(f1, 4)
        )

        # Log parameters
        mlflow.log_param(
            "model",
            "Logistic Regression"
        )

        mlflow.log_param(
            "test_size",
            0.20
        )

        mlflow.log_param(
            "random_state",
            42
        )

        mlflow.log_param(
            "max_iter",
            1000
        )

        mlflow.log_param(
            "dataset",
            DATA_PATH
        )

        mlflow.log_param(
            "preprocessing",
            "Median Imputation + StandardScaler + OneHotEncoder"
        )

        # Log metrics
        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        # Create outputs folder
        os.makedirs(
            OUTPUT_DIR,
            exist_ok=True
        )

        # Save results
        results = {
            "experiment": 4,
            "model": "Logistic Regression",
            "dataset": DATA_PATH,
            "training_samples": len(X_train),
            "testing_samples": len(X_test),
            "test_size": 0.20,
            "random_state": 42,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        output_file = os.path.join(
            OUTPUT_DIR,
            "experiment_4_results.json"
        )

        with open(
            output_file,
            "w"
        ) as file:

            json.dump(
                results,
                file,
                indent=4
            )

        # Log results JSON
        mlflow.log_artifact(
            output_file
        )

        # Save trained model as PKL
        model_file = os.path.join(
            OUTPUT_DIR,
            "experiment_4_logistic_regression.pkl"
        )

        joblib.dump(
            model,
            model_file
        )

        # Log PKL model as MLflow artifact
        mlflow.log_artifact(
            model_file,
            artifact_path="model"
        )

        print("\nModel artifact saved:")
        print(model_file)

        # Get MLflow run ID
        run_id = (
            mlflow.active_run()
            .info
            .run_id
        )

        print("\nMLflow Run ID:")
        print(run_id)

        print("\nResults saved:")
        print(output_file)

    print("\n========================================")
    print("Experiment 4 completed successfully.")
    print("========================================")


if __name__ == "__main__":

    run_experiment()