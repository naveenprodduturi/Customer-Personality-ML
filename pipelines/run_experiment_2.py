import pandas as pd
import numpy as np
import os
import json
import csv
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

DATA_PATH = "data/raw/marketing_campaign.csv"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"


# ============================================================
# LOAD DATASET
# ============================================================

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

    # Clean target column
    data["Response"] = (
        data["Response"]
        .astype(str)
        .str.strip()
        .str.strip('"')
        .str.strip("'")
        .astype(int)
    )

    return data


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(data):

    # Remove ID and target
    X = data.drop(
        columns=["ID", "Response"]
    )

    y = data["Response"]

    # Convert date column
    X["Dt_Customer"] = pd.to_datetime(
        X["Dt_Customer"],
        dayfirst=True,
        errors="coerce"
    )

    # Create date-based features
    X["Customer_Year"] = X["Dt_Customer"].dt.year
    X["Customer_Month"] = X["Dt_Customer"].dt.month
    X["Customer_Day"] = X["Dt_Customer"].dt.day

    # Remove original date column
    X = X.drop(
        columns=["Dt_Customer"]
    )

    return X, y


# ============================================================
# CREATE PREPROCESSING PIPELINE
# ============================================================

def create_preprocessor(X):

    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "str"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
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

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return preprocessor


# ============================================================
# CREATE MODELS
# ============================================================

def create_models():

    models = {

        "Logistic_Regression":
            LogisticRegression(
                max_iter=1000,
                random_state=42
            ),

        "Decision_Tree":
            DecisionTreeClassifier(
                random_state=42
            ),

        "Random_Forest":
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
    }

    return models


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

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

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": matrix.tolist()
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print(
        "\n========================================"
    )

    print(
        "EXPERIMENT 2 - BASELINE MODEL EVALUATION"
    )

    print(
        "========================================\n"
    )

    # Load dataset
    print("Loading dataset...")

    data = load_dataset()

    print(
        "Dataset shape:",
        data.shape
    )

    # Prepare data
    X, y = prepare_data(
        data
    )

    print(
        "Feature shape:",
        X.shape
    )

    print(
        "Target shape:",
        y.shape
    )

    # Train-test split
    print(
        "\nCreating reproducible train-test split..."
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(
        "Training samples:",
        X_train.shape[0]
    )

    print(
        "Testing samples:",
        X_test.shape[0]
    )

    # Create preprocessor
    preprocessor = create_preprocessor(
        X_train
    )

    # Create models
    models = create_models()

    results = {}

    # Create directories
    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # Train each model
    for model_name, classifier in models.items():

        print(
            f"\n----------------------------------------"
        )

        print(
            f"Training {model_name}..."
        )

        # Create complete pipeline
        model_pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "classifier",
                    classifier
                )
            ]
        )

        # Train
        model_pipeline.fit(
            X_train,
            y_train
        )

        print(
            f"{model_name} training completed."
        )

        # Evaluate
        evaluation = evaluate_model(
            model_pipeline,
            X_test,
            y_test
        )

        results[model_name] = evaluation

        # Save model
        model_path = os.path.join(
            MODEL_DIR,
            f"{model_name.lower()}.pkl"
        )

        joblib.dump(
            model_pipeline,
            model_path
        )

        print(
            "Model saved:",
            model_path
        )

        # Print metrics
        print(
            f"Accuracy  : {evaluation['accuracy']:.4f}"
        )

        print(
            f"Precision : {evaluation['precision']:.4f}"
        )

        print(
            f"Recall    : {evaluation['recall']:.4f}"
        )

        print(
            f"F1 Score  : {evaluation['f1_score']:.4f}"
        )

        print(
            "Confusion Matrix:"
        )

        print(
            np.array(
                evaluation["confusion_matrix"]
            )
        )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "MODEL COMPARISON"
    )

    print(
        "========================================"
    )

    for model_name, metrics in results.items():

        print(
            f"\n{model_name}"
        )

        print(
            f"Accuracy  : {metrics['accuracy']:.4f}"
        )

        print(
            f"Precision : {metrics['precision']:.4f}"
        )

        print(
            f"Recall    : {metrics['recall']:.4f}"
        )

        print(
            f"F1 Score  : {metrics['f1_score']:.4f}"
        )

    # ========================================================
    # BEST MODEL
    # ========================================================

    best_model = max(
        results,
        key=lambda name:
        results[name]["f1_score"]
    )

    print(
        "\nBest model based on F1 Score:"
    )

    print(
        best_model
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    experiment_results = {

        "experiment": "Experiment 2",

        "dataset": "Customer Personality Analysis",

        "dataset_shape": [
            int(data.shape[0]),
            int(data.shape[1])
        ],

        "test_size": 0.20,

        "random_state": 42,

        "models": results,

        "best_model": best_model,

        "model_version": "1.0"
    }

    results_path = os.path.join(
        OUTPUT_DIR,
        "experiment_2_results.json"
    )

    with open(
        results_path,
        "w"
    ) as file:

        json.dump(
            experiment_results,
            file,
            indent=4
        )

    print(
        "\nExperiment 2 results saved:"
    )

    print(
        results_path
    )

    print(
        "\nExperiment 2 completed successfully."
    )


if __name__ == "__main__":

    main()