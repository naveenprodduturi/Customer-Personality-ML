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


# ========================================
# EXPERIMENT 7 - FINAL MODEL AND
# UNSEEN CUSTOMER PREDICTION
# ========================================

DATA_PATH = "data/raw/marketing_campaign.csv"
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"


def load_dataset():

    print("Loading dataset...")

    data = pd.read_csv(
        DATA_PATH,
        sep="\t",
        engine="python"
    )

    # Fix single-column TSV format
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


def run_experiment():

    print("\n" + "=" * 55)
    print("EXPERIMENT 7 - FINAL MODEL AND UNSEEN PREDICTION")
    print("=" * 55)

    # Load dataset
    data = load_dataset()

    # Prepare data
    X, y, preprocessing = prepare_data(
        data
    )

    # Split for final verification
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # Final model
    final_model = Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessing
            ),
            (
                "classifier",
                LogisticRegression(
                    C=1,
                    solver="liblinear",
                    max_iter=1000,
                    random_state=42
                )
            )
        ]
    )

    print("\nTraining final Logistic Regression model...")

    final_model.fit(
        X_train,
        y_train
    )

    print("Final model training completed.")

    # Test performance
    test_predictions = final_model.predict(
        X_test
    )

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score
    )

    accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    precision = precision_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    print("\n----------------------------------------")
    print("FINAL MODEL PERFORMANCE")
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

    # ========================================
    # UNSEEN CUSTOMER
    # ========================================

    print("\n----------------------------------------")
    print("UNSEEN CUSTOMER PREDICTION")
    print("----------------------------------------")

    unseen_customer = pd.DataFrame(
        [
            {
                "Year_Birth": 1975,
                "Education": "Graduation",
                "Marital_Status": "Married",
                "Income": 50000,
                "Kidhome": 1,
                "Teenhome": 0,
                "Recency": 20,
                "MntWines": 500,
                "MntFruits": 50,
                "MntMeatProducts": 300,
                "MntFishProducts": 70,
                "MntSweetProducts": 60,
                "MntGoldProds": 80,
                "NumDealsPurchases": 3,
                "NumWebPurchases": 5,
                "NumCatalogPurchases": 4,
                "NumStorePurchases": 6,
                "NumWebVisitsMonth": 5,
                "AcceptedCmp3": 0,
                "AcceptedCmp4": 0,
                "AcceptedCmp5": 0,
                "AcceptedCmp1": 0,
                "AcceptedCmp2": 0,
                "Complain": 0,
                "Z_CostContact": 3,
                "Z_Revenue": 11
            }
        ]
    )

    # Prediction
    prediction = final_model.predict(
        unseen_customer
    )[0]

    probability = final_model.predict_proba(
        unseen_customer
    )[0]

    response_probability = probability[1] * 100

    if prediction == 1:

        prediction_text = "Likely to respond"

    else:

        prediction_text = "Unlikely to respond"

    print(
        "Prediction:",
        prediction_text
    )

    print(
        "Response probability:",
        round(
            response_probability,
            2
        ),
        "%"
    )

    # ========================================
    # SAVE FINAL MODEL
    # ========================================

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    model_file = os.path.join(
        MODEL_DIR,
        "final_model.pkl"
    )

    joblib.dump(
        final_model,
        model_file
    )

    print("\nFinal model saved:")
    print(model_file)

    # ========================================
    # SAVE RESULTS
    # ========================================

    results = {

        "experiment": 7,

        "model": "Final Logistic Regression",

        "hyperparameters": {
            "C": 1,
            "solver": "liblinear",
            "max_iter": 1000,
            "random_state": 42
        },

        "test_accuracy": accuracy,

        "test_precision": precision,

        "test_recall": recall,

        "test_f1": f1,

        "unseen_prediction": prediction_text,

        "unseen_prediction_class": int(prediction),

        "response_probability_percent":
            response_probability
    }

    output_file = os.path.join(
        OUTPUT_DIR,
        "experiment_7_results.json"
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

    print("\nResults saved:")
    print(output_file)

    print("\n" + "=" * 55)
    print("EXPERIMENT 7 COMPLETED SUCCESSFULLY")
    print("=" * 55)


if __name__ == "__main__":

    run_experiment()