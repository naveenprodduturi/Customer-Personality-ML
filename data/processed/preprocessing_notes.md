# Customer Personality Dataset - Preprocessing Notes

## 1. Dataset

Dataset Name: Customer Personality Analysis

Source File: marketing_campaign.csv

Number of Records: 2240

Number of Columns: 29

Target Variable: Response

Task Type: Binary Classification


## 2. Dataset Loading

The dataset is loaded from the raw data directory.

The original dataset uses tab-separated values, so the tab separator is used while reading the file.


## 3. Column Name Cleaning

Column names are cleaned by removing:

- Leading and trailing spaces
- Unwanted quotation marks

This ensures that column names can be accessed consistently during preprocessing and model development.


## 4. Target Variable Cleaning

The Response column is the target variable.

The original values contained quotation marks. These quotation marks are removed and the values are converted into integer values.

Target classes:

- 0 = Customer did not respond
- 1 = Customer responded


## 5. Missing Value Analysis

The dataset contains 24 missing values in the Income column.

No other columns contain missing values.

Missing Income values are handled during the machine learning preprocessing pipeline using median imputation.


## 6. Duplicate Analysis

The dataset contains zero duplicate records.

Therefore, no duplicate records were removed.


## 7. Categorical Features

The main categorical features are:

- Education
- Marital_Status

These categorical variables are handled using imputation and one-hot encoding in the machine learning pipeline.


## 8. Numerical Features

Numerical features include:

- Year_Birth
- Income
- Kidhome
- Teenhome
- Recency
- MntWines
- MntFruits
- MntMeatProducts
- MntFishProducts
- MntSweetProducts
- MntGoldProds
- NumDealsPurchases
- NumWebPurchases
- NumCatalogPurchases
- NumStorePurchases
- NumWebVisitsMonth
- AcceptedCmp3
- AcceptedCmp4
- AcceptedCmp5
- AcceptedCmp1
- AcceptedCmp2
- Complain
- Z_CostContact
- Z_Revenue


## 9. Identifier Removal

The ID column is an identifier and does not represent a predictive customer feature.

Therefore, ID is removed before model training.


## 10. Date Processing

Dt_Customer contains customer registration dates.

The date is converted into separate numerical features:

- Customer_Year
- Customer_Month
- Customer_Day

The original Dt_Customer column is then removed.


## 11. Numerical Preprocessing

Numerical features are processed using:

1. Median imputation for missing numerical values.
2. StandardScaler for feature scaling.


## 12. Categorical Preprocessing

Categorical features are processed using:

1. Most-frequent-value imputation.
2. One-hot encoding.

The encoder uses handle_unknown="ignore" so that unseen categories do not cause errors during prediction.


## 13. Dataset Splitting

The dataset is divided into training and testing data.

Training data:

80%

Testing data:

20%

The split uses random_state = 42 to make the experiment reproducible.

Stratified splitting is used to preserve the distribution of the target classes.


## 14. Target Distribution

The dataset contains:

Response = 0: 1906 records

Response = 1: 334 records

The target classes are therefore imbalanced, with substantially more non-response records than response records.


## 15. Reproducibility

The following practices are used to support reproducibility:

- Fixed random state of 42
- Consistent preprocessing pipeline
- Saved dataset metadata
- Saved trained model
- Separate raw and processed data directories


## 16. Output

Dataset metadata is stored in:

data/processed/dataset_metadata.json

The trained Logistic Regression model is stored in:

models/logistic_regression_model.pkl

Evaluation results are stored in:

outputs/evaluation_results.json
## 17. Reproducible Train-Test Split

The dataset is divided into training and testing subsets using the following configuration:

- Training data: 80%
- Testing data: 20%
- Random state: 42
- Stratification: Enabled
- Target variable: Response

Using a fixed random state ensures that the same train-test split can be reproduced in future executions.

Stratification preserves the relative distribution of the Response classes in both training and testing datasets.