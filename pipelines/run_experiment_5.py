import os
import json
import joblib
import mlflow
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
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
# EXPERIMENT 5 - HYPERPARAMETER TUNING
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

    # Fix one-column TSV format
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

    # Clean target
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

    return data


def prepare_data(data):

    target = "Response"

    # Remove ID and date
    data = data.drop(
        columns=[
            column
            for column in ["ID", "Dt_Customer"]
            if column in data.columns
        ]
    )

    X = data.drop(
        columns=[target]
    )

    y = data[target].astype(int)

    categorical_columns = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    numerical_columns = X.select_dtypes(
        exclude=["object", "string"]
    ).columns.tolist()

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
    print("EXPERIMENT 5 - HYPERPARAMETER TUNING")
    print("=" * 50)

    data = load_dataset()

    X, y, preprocessing = prepare_data(
        data
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # Base model
    pipeline = Pipeline(
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

    # Hyperparameter grid
    param_grid = {
        "classifier__C": [
            0.01,
            0.1,
            1,
            10
        ],

        "classifier__solver": [
            "liblinear",
            "lbfgs"
        ]
    }

    print("\nStarting GridSearchCV...")

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(
        X_train,
        y_train
    )

    print("\n----------------------------------------")
    print("BEST HYPERPARAMETERS")
    print("----------------------------------------")

    print(
        "Best parameters:",
        grid_search.best_params_
    )

    print(
        "Best CV F1 Score:",
        round(
            grid_search.best_score_,
            4
        )
    )

    # Best model
    best_model = grid_search.best_estimator_

    predictions = best_model.predict(
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

    print("\n----------------------------------------")
    print("TUNED MODEL TEST RESULTS")
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

    # MLflow
    mlflow.set_experiment(
        "Customer_Personality_Experiment_5"
    )

    with mlflow.start_run(
        run_name="Logistic_Regression_Hyperparameter_Tuning"
    ):

        mlflow.log_param(
            "model",
            "Logistic Regression"
        )

        mlflow.log_param(
            "tuning_method",
            "GridSearchCV"
        )

        mlflow.log_param(
            "cross_validation",
            5
        )

        mlflow.log_param(
            "scoring",
            "F1"
        )

        mlflow.log_params(
            grid_search.best_params_
        )

        mlflow.log_metric(
            "best_cv_f1",
            grid_search.best_score_
        )

        mlflow.log_metric(
            "test_accuracy",
            accuracy
        )

        mlflow.log_metric(
            "test_precision",
            precision
        )

        mlflow.log_metric(
            "test_recall",
            recall
        )

        mlflow.log_metric(
            "test_f1",
            f1
        )

        os.makedirs(
            OUTPUT_DIR,
            exist_ok=True
        )

        results = {
            "experiment": 5,
            "model": "Logistic Regression",
            "method": "GridSearchCV",
            "cv": 5,
            "scoring": "f1",
            "best_parameters": grid_search.best_params_,
            "best_cv_f1": grid_search.best_score_,
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1
        }

        output_file = os.path.join(
            OUTPUT_DIR,
            "experiment_5_results.json"
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

        mlflow.log_artifact(
            output_file
        )

        # Save tuned model
        model_file = os.path.join(
            OUTPUT_DIR,
            "experiment_5_tuned_model.pkl"
        )

        joblib.dump(
            best_model,
            model_file
        )

        mlflow.log_artifact(
            model_file,
            artifact_path="model"
        )

        print("\nBest model saved:")
        print(model_file)

        print("\nMLflow Run ID:")
        print(
            mlflow.active_run().info.run_id
        )

    print("\n========================================")
    print("Experiment 5 completed successfully.")
    print("========================================")


if __name__ == "__main__":

    run_experiment()