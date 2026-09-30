import os
import json
import joblib
import pandas as pd

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


# ========================================
# EXPERIMENT 6 - MODEL EVALUATION
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

    # Fix single-column TSV
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

    # Clean columns
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

    # Target
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


def evaluate_model(
    name,
    model,
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\n----------------------------------------")
    print("Evaluating:", name)
    print("----------------------------------------")

    model.fit(
        X_train,
        y_train
    )

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

    cm = confusion_matrix(
        y_test,
        predictions
    )

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

    print("Confusion Matrix:")
    print(cm)

    return {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm.tolist()
    }, model


def run_experiment():

    print("\n" + "=" * 50)
    print("EXPERIMENT 6 - MODEL EVALUATION AND COMPARISON")
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

    # ----------------------------------------
    # Models
    # ----------------------------------------

    models = {

        "Logistic_Regression":
            LogisticRegression(
                max_iter=1000,
                random_state=42
            ),

        "Decision_Tree":
            DecisionTreeClassifier(
                max_depth=5,
                random_state=42
            ),

        "Random_Forest":
            RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
    }

    results = []

    trained_models = {}

    for name, classifier in models.items():

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessing",
                    preprocessing
                ),
                (
                    "classifier",
                    classifier
                )
            ]
        )

        result, trained_model = evaluate_model(
            name,
            pipeline,
            X_train,
            X_test,
            y_train,
            y_test
        )

        results.append(result)

        trained_models[name] = trained_model

    # ----------------------------------------
    # Comparison
    # ----------------------------------------

    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)

    for result in results:

        print(
            "\n",
            result["model"]
        )

        print(
            "Accuracy  :",
            round(result["accuracy"], 4)
        )

        print(
            "Precision :",
            round(result["precision"], 4)
        )

        print(
            "Recall    :",
            round(result["recall"], 4)
        )

        print(
            "F1 Score  :",
            round(result["f1_score"], 4)
        )

    # Best model based on F1
    best_result = max(
        results,
        key=lambda x: x["f1_score"]
    )

    best_model_name = best_result["model"]

    print("\n" + "=" * 50)
    print("BEST MODEL")
    print("=" * 50)

    print(
        "Best model based on F1 Score:",
        best_model_name
    )

    print(
        "Best F1 Score:",
        round(
            best_result["f1_score"],
            4
        )
    )

    # ----------------------------------------
    # Save results
    # ----------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        "experiment_6_results.json"
    )

    final_results = {
        "experiment": 6,
        "evaluation_metric": "F1 Score",
        "results": results,
        "best_model": best_model_name,
        "best_f1_score": best_result["f1_score"]
    }

    with open(
        output_file,
        "w"
    ) as file:

        json.dump(
            final_results,
            file,
            indent=4
        )

    # Save best model
    best_model_file = os.path.join(
        OUTPUT_DIR,
        "experiment_6_best_model.pkl"
    )

    joblib.dump(
        trained_models[best_model_name],
        best_model_file
    )

    print("\nResults saved:")
    print(output_file)

    print("\nBest model saved:")
    print(best_model_file)

    print("\n========================================")
    print("Experiment 6 completed successfully.")
    print("========================================")


if __name__ == "__main__":

    run_experiment()