# Experiment 2 - Baseline Model Comparison and Analysis

## Objective

The objective of Experiment 2 is to develop baseline classification models for predicting customer campaign responses and compare their performance using standard evaluation metrics.

The following models were developed:

1. Logistic Regression
2. Decision Tree
3. Random Forest

## Dataset Split

The Customer Personality Analysis dataset contains 2240 records and 29 columns.

The dataset was divided using:

- Training data: 80%
- Testing data: 20%
- Random state: 42
- Stratification: Enabled

Training samples: 1792

Testing samples: 448


## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8839 | 0.7143 | 0.3731 | 0.4902 |
| Decision Tree | 0.8170 | 0.3846 | 0.3731 | 0.3788 |
| Random Forest | 0.8862 | 0.8077 | 0.3134 | 0.4516 |


## Confusion Matrix Results

### Logistic Regression

Confusion Matrix:

[[371, 10],
 [42, 25]]

The model correctly classified 371 customers with Response = 0 and 25 customers with Response = 1.

It incorrectly classified 10 negative samples as positive and 42 positive samples as negative.


### Decision Tree

Confusion Matrix:

[[341, 40],
 [42, 25]]

The Decision Tree correctly classified 341 negative samples and 25 positive samples.

It produced 40 false positives and 42 false negatives.


### Random Forest

Confusion Matrix:

[[376, 5],
 [46, 21]]

The Random Forest correctly classified 376 negative samples and 21 positive samples.

It produced 5 false positives and 46 false negatives.


## Model Comparison

Random Forest achieved the highest accuracy of 0.8862 and the highest precision of 0.8077.

Logistic Regression achieved the highest recall among the three models at 0.3731 and the highest F1 Score of 0.4902.

Decision Tree produced the lowest accuracy, precision, and F1 Score among the three models.

Although Random Forest achieved slightly higher accuracy than Logistic Regression, Logistic Regression achieved the highest F1 Score.

Therefore, Logistic Regression was selected as the best baseline model based on F1 Score.


## Analysis of Poor Predictions

The models show difficulty in correctly identifying customers who responded to the campaign.

The dataset contains many more non-response cases than response cases. This class imbalance affects the ability of the baseline models to identify the minority positive class.

The confusion matrices show a relatively high number of false negatives.

For Logistic Regression, 42 positive customers were incorrectly predicted as negative.

For Decision Tree, 42 positive customers were incorrectly predicted as negative.

For Random Forest, 46 positive customers were incorrectly predicted as negative.

This indicates that improving minority-class prediction should be considered in later experiments or future model improvements.


## Final Baseline Model

Based on the F1 Score, Logistic Regression was selected as the best baseline model.

Best baseline model:

Logistic Regression

Accuracy: 0.8839

Precision: 0.7143

Recall: 0.3731

F1 Score: 0.4902


## Model Serialization

The trained models were serialized using Joblib.

Saved models include:

- models/logistic_regression.pkl
- models/decision_tree.pkl
- models/random_forest.pkl

The Experiment 2 evaluation results are stored in:

outputs/experiment_2_results.json

Model version:

1.0