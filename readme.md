# 💳 Credit Card Fraud Detection System

**Machine Learning · Python · Streamlit**

🚀 **Live Demo:** https://credit-card-fraud-detection-xngn.onrender.com/

A machine-learning web application that analyzes transaction data
uploaded as a CSV and flags transactions predicted to be potentially
fraudulent.

> **Disclaimer:** This is an educational/demo project. A model
> prediction is not proof of fraud and should not be used as the sole
> basis for real financial decisions.

## Overview

Credit card fraud detection is a binary classification task. This
application lets a user upload transaction data, run predictions with a
saved model, review summary counts and row-level results, and download a
CSV report.

## Features

-   Upload a transaction CSV through a web interface.
-   Validate required feature columns and numeric values.
-   Predict a class for each transaction using a previously trained
    model.
-   Show estimated fraud probability when supported by the classifier.
-   Display total transactions, predicted fraud count, and predicted
    legitimate count.
-   Review predictions in a table and download the result as CSV.
-   Access the app online through the deployed demo.

## Technology Stack

  Technology     Purpose
  -------------- -----------------------------------------
  Python         Application and ML programming
  pandas         CSV reading and tabular data processing
  NumPy          Numerical operations and value checks
  scikit-learn   Machine-learning model and utilities
  joblib         Saving and loading model artifacts
  Streamlit      Interactive web interface
  GitHub         Source-code hosting
  Render         Hosting for the live application


## Dataset

The commonly used public credit-card fraud dataset contains these
fields:

  -----------------------------------------------------------------------
  Column                              Description
  ----------------------------------- -----------------------------------
  `Time`                              Elapsed time from the first
                                      transaction in the dataset

  `V1`--`V28`                         Anonymized numerical features,
                                      commonly transformed using PCA

  `Amount`                            Transaction amount

  `Class`                             Label in labelled data: `0` =
                                      legitimate, `1` = fraud
  -----------------------------------------------------------------------

A widely used version contains 284,807 transactions and 492 fraud cases.
Verify the exact data used by your training process before quoting these
numbers as project-specific statistics. The anonymized `V1`--`V28`
features do not directly reveal their original business meanings.

## How It Works

1.  The user uploads a CSV file.
2.  The app checks that required feature columns exist and values are
    valid.
3.  It selects features in the order expected by the model.
4.  The saved model predicts a class for each transaction.
5.  If supported, the app retrieves an estimated probability for the
    fraud class.
6.  The dashboard displays summary counts and a prediction table.
7.  The user can download the results as a CSV report.

### Model

The project context refers to a Random Forest classifier and class
weighting; verify the actual training notebook before claiming these as
exact implementation details. Random Forest combines predictions from
multiple decision trees. Class weighting can help a model pay more
attention to a minority class, but it does not guarantee good fraud
detection.

### Training vs. Inference

Training learns patterns from labelled historical data. The deployed app
normally loads a previously trained model and performs **inference** on
uploaded data; it does not retrain the model for every upload.

## Evaluation Metrics

Fraud datasets are often imbalanced, so accuracy alone can be
misleading. Useful metrics include:

-   **Precision:** Among transactions predicted as fraud, the proportion
    that were actually fraud.
-   **Recall:** Among actual fraud cases, the proportion detected.
-   **F1-score:** Harmonic mean of precision and recall.
-   **Confusion matrix:** Counts true positives, true negatives, false
    positives, and false negatives.
-   **PR-AUC:** Summarizes the precision-recall trade-off across
    thresholds.

Do not report a performance score unless it comes from a valid
independent labelled test set. If an uploaded file has no actual labels,
the app can summarize predictions but cannot calculate accuracy for
those rows.

## Project Structure

Typical structure:

``` text
credit-card-fraud-detection/
├── app.py
├── fraud_model.pkl
├── feature_columns.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

The repository may contain additional files. Avoid committing large raw
datasets or sensitive transaction information unless there is a clear,
safe reason.

## Run Locally

### 1. Clone the repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

Replace the placeholders with your actual repository URL and folder
name.

### 2. Create a virtual environment (recommended)

**Windows PowerShell:**

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

Ensure `requirements.txt` pins a scikit-learn version compatible with
the version used to save `fraud_model.pkl`. Serialized models may fail
to load across incompatible versions.

### 4. Run the app

``` bash
python -m streamlit run app.py
```

The app normally opens at `http://localhost:8501`. Ensure the model and
feature-column files are present at the paths expected by `app.py`.

## Input Requirements

-   Upload a CSV file.
-   Include every feature column required by the trained model.
-   Use the same feature definitions and preprocessing expected during
    training.
-   Supply numeric, finite values unless missing values are explicitly
    supported.
-   Do not upload sensitive real customer data to a public demo.

## Security and Responsible Use

-   Never commit cardholder data, personal information, credentials, or
    secrets.
-   Load only trusted `.pkl`/joblib artifacts; untrusted serialized
    files can execute code.
-   Treat predictions as indicators for review, not definitive proof.
-   A production banking system would require strong security, privacy
    controls, monitoring, validation, and auditability.
-   Fraud patterns can change over time, so model performance should be
    monitored.

## Limitations

-   The app processes uploaded CSVs; it is not connected to a live bank
    or payment processor.
-   False positives and false negatives are possible.
-   Anonymized features limit direct interpretation.
-   Probability outputs are estimates and may not be calibrated.
-   Reliable evaluation requires independent labelled test data and
    metrics suited to class imbalance.

## Future Improvements

-   Compare Random Forest with Logistic Regression and gradient-boosting
    models.
-   Tune the decision threshold based on precision-recall trade-offs and
    the costs of missed fraud and false alarms.
-   Report a confusion matrix, precision, recall, F1-score, and PR-AUC
    on an independent test set.
-   Add explainability, secure authentication, upload limits, and
    privacy protections.
-   Monitor data drift and establish a controlled retraining process.

## Contributors


-   **Project team:** Data Defenders
-   **Repository:** https://github.com/Taranjeet07/credit-card-fraud-detection



## Acknowledgements

This project demonstrates a machine-learning classification workflow and
a web interface for reviewing transaction predictions.
