import pandas as pd
import joblib
import os
import csv

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression


# File paths
DATA_PATH = "data/raw/marketing_campaign.csv"
MODEL_PATH = "models/logistic_regression_model.pkl"


def load_dataset():
    """
    Load and clean the Customer Personality dataset.
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


def prepare_data(data):
    """
    Prepare features and target variable.
    """

    # Remove ID because it is only an identifier
    X = data.drop(
        columns=["ID", "Response"]
    )

    y = data["Response"]

    # Convert date column into useful numerical features
    X["Dt_Customer"] = pd.to_datetime(
        X["Dt_Customer"],
        dayfirst=True,
        errors="coerce"
    )

    X["Customer_Year"] = X["Dt_Customer"].dt.year
    X["Customer_Month"] = X["Dt_Customer"].dt.month
    X["Customer_Day"] = X["Dt_Customer"].dt.day

    # Remove original date column
    X = X.drop(
        columns=["Dt_Customer"]
    )

    return X, y


def build_pipeline(X):
    """
    Build preprocessing and Logistic Regression pipeline.
    """

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
                SimpleImputer(strategy="median")
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
                SimpleImputer(strategy="most_frequent")
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
                "num",
                numeric_pipeline,
                numeric_features
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    model_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000
                )
            )
        ]
    )

    return model_pipeline


def main():

    print("Loading dataset...")

    data = load_dataset()

    print("Dataset shape:", data.shape)

    X, y = prepare_data(data)

    print("Features:", X.shape)
    print("Target:", y.shape)

    print("\nSplitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])

    print("\nBuilding Logistic Regression pipeline...")

    model = build_pipeline(X_train)

    print("Training model...")

    model.fit(
        X_train,
        y_train
    )

    print("Model training completed.")

    # Create models directory
    os.makedirs(
        "models",
        exist_ok=True
    )

    # Save trained model
    joblib.dump(
        model,
        MODEL_PATH
    )

    print("\nModel saved successfully.")

    print(
        "Model location:",
        MODEL_PATH
    )


if __name__ == "__main__":
    main()