<<<<<<< HEAD
VaultIQ — AI Loan Intelligence
=======
# VaultIQ — AI Loan Intelligence
>>>>>>> 25a3ce7 (Improve The README File)

<p align="center">
  <strong>End-to-end Machine Learning system for loan approval prediction</strong><br>
  <sub>From data exploration and model optimization to an interactive Streamlit application</sub>
</p>

<p align="center">
  <a href="https://github.com/amir-4/Loan-Approval-Classification">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github" alt="GitHub Repository">
  </a>
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/CatBoost-Classifier-F2C811?style=for-the-badge" alt="CatBoost">
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly">
</p>

<<<<<<< HEAD
Project Overview

VaultIQ is an end-to-end Machine Learning project for predicting whether a loan application is likely to be approved or rejected based on applicant, financial, employment, loan, and credit-history information.

The project was built as a complete ML workflow rather than a standalone notebook. It covers data understanding, data quality checks, preprocessing, exploratory analysis, model benchmarking, hyperparameter optimization, evaluation, model persistence, and deployment through an interactive Streamlit application.

The final deployed application uses an optimized CatBoostClassifier and reports 94% accuracy on the held-out test set.

Important: VaultIQ is a portfolio/educational project and a decision-support demonstration. It is not intended to replace professional underwriting, regulatory checks, or real-world lending decisions.

What VaultIQ Does

The application allows a user to enter an applicant's information and receive an immediate prediction together with model insights.

Core capabilities

Loan approval / rejection prediction

Approval probability visualization

Interactive probability charts

Model feature-importance analysis

Top feature-driver visualization

Automatic calculation of loan-to-income ratio

Applicant profile summary

Modern finance-focused Streamlit interface

Native CatBoost model loading for deployment

Machine Learning Workflow

Raw Loan Data
     ↓
Data Understanding & Quality Checks
     ↓
Exploratory Data Analysis
     ↓
Feature Preparation
     ↓
Train / Test Split
     ↓
Model Benchmarking
     ↓
Hyperparameter Optimization
     ↓
Final CatBoost Model
     ↓
Evaluation on Held-Out Test Data
     ↓
Model Serialization (.cbm)
     ↓
Streamlit Deployment

Dataset

The project uses the Loan Approval Classification Dataset from Kaggle:

Source: https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data

The notebook contains 45,000 records and 14 columns: 13 input features plus the target variable loan_status.

Target

Value

Meaning

1

Approved

0

Rejected

Input Features

Feature

Description

Type

person_age

Applicant age

Numeric

person_gender

Applicant gender

Categorical

person_education

Highest education level

Categorical

person_income

Annual income

Numeric

person_emp_exp

Employment experience in years

Numeric

person_home_ownership

Home ownership status

Categorical

loan_amnt

Requested loan amount

Numeric

loan_intent

Intended purpose of the loan

Categorical

loan_int_rate

Loan interest rate

Numeric

loan_percent_income

Loan amount as a percentage of annual income

Numeric

cb_person_cred_hist_length

Credit history length

Numeric

credit_score

Applicant credit score

Numeric

previous_loan_defaults_on_file

Previous loan default indicator

Categorical

Data Preparation

The notebook includes data-quality and preprocessing steps such as:

Dataset inspection and schema validation

Missing-value checks

Duplicate checks

Descriptive statistics

Numerical and categorical feature analysis

Train/test splitting

Encoding of categorical variables for the traditional scikit-learn pipeline

Numerical feature scaling where required

Stratified cross-validation for model selection and optimization

For the final CatBoost workflow, the original categorical features are supplied directly to CatBoost using its native categorical-feature handling.

Models Evaluated

Multiple classification algorithms were considered during model benchmarking, including:

Logistic Regression

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

Gaussian Naive Bayes

Decision Tree

Random Forest

XGBoost

LightGBM

CatBoost

The purpose of benchmarking was to compare different approaches under a consistent evaluation workflow and identify strong candidates for further tuning.

CatBoost Optimization
=======
---

## Project Overview

**VaultIQ** is an end-to-end Machine Learning project for predicting whether a loan application is likely to be **approved or rejected** based on applicant, financial, employment, loan, and credit-history information.

The project was built as a complete ML workflow rather than a standalone notebook. It covers **data understanding, data quality checks, preprocessing, exploratory analysis, model benchmarking, hyperparameter optimization, evaluation, model persistence, and deployment** through an interactive Streamlit application.

The final deployed application uses an optimized **CatBoostClassifier** and reports **94% accuracy on the held-out test set**.

> **Important:** VaultIQ is a portfolio/educational project and a decision-support demonstration. It is not intended to replace professional underwriting, regulatory checks, or real-world lending decisions.

---

## What VaultIQ Does

The application allows a user to enter an applicant's information and receive an immediate prediction together with model insights.

### Core capabilities

- Loan approval / rejection prediction
- Approval probability visualization
- Interactive probability charts
- Model feature-importance analysis
- Top feature-driver visualization
- Automatic calculation of loan-to-income ratio
- Applicant profile summary
- Modern finance-focused Streamlit interface
- Native CatBoost model loading for deployment

---

## Machine Learning Workflow

```text
Raw Loan Data
     ↓
Data Understanding & Quality Checks
     ↓
Exploratory Data Analysis
     ↓
Feature Preparation
     ↓
Train / Test Split
     ↓
Model Benchmarking
     ↓
Hyperparameter Optimization
     ↓
Final CatBoost Model
     ↓
Evaluation on Held-Out Test Data
     ↓
Model Serialization (.cbm)
     ↓
Streamlit Deployment
```

---

## Dataset

The project uses the **Loan Approval Classification Dataset** from Kaggle:

**Source:** https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data

The notebook contains **45,000 records** and **14 columns**: 13 input features plus the target variable `loan_status`.

### Target

| Value | Meaning |
|---:|---|
| `1` | Approved |
| `0` | Rejected |

### Input Features

| Feature | Description | Type |
|---|---|---|
| `person_age` | Applicant age | Numeric |
| `person_gender` | Applicant gender | Categorical |
| `person_education` | Highest education level | Categorical |
| `person_income` | Annual income | Numeric |
| `person_emp_exp` | Employment experience in years | Numeric |
| `person_home_ownership` | Home ownership status | Categorical |
| `loan_amnt` | Requested loan amount | Numeric |
| `loan_intent` | Intended purpose of the loan | Categorical |
| `loan_int_rate` | Loan interest rate | Numeric |
| `loan_percent_income` | Loan amount as a percentage of annual income | Numeric |
| `cb_person_cred_hist_length` | Credit history length | Numeric |
| `credit_score` | Applicant credit score | Numeric |
| `previous_loan_defaults_on_file` | Previous loan default indicator | Categorical |

---

## Data Preparation

The notebook includes data-quality and preprocessing steps such as:

- Dataset inspection and schema validation
- Missing-value checks
- Duplicate checks
- Descriptive statistics
- Numerical and categorical feature analysis
- Train/test splitting
- Encoding of categorical variables for the traditional scikit-learn pipeline
- Numerical feature scaling where required
- Stratified cross-validation for model selection and optimization

For the final CatBoost workflow, the original categorical features are supplied directly to CatBoost using its native categorical-feature handling.

---

## Models Evaluated

Multiple classification algorithms were considered during model benchmarking, including:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Gaussian Naive Bayes
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- CatBoost

The purpose of benchmarking was to compare different approaches under a consistent evaluation workflow and identify strong candidates for further tuning.

---

## CatBoost Optimization
>>>>>>> 25a3ce7 (Improve The README File)

CatBoost was selected for the final deployed model because it provides native support for categorical features and is well suited to gradient-boosting classification tasks.

The optimization stage uses:

<<<<<<< HEAD
RandomizedSearchCV

Stratified 5-Fold Cross-Validation

Accuracy as the primary search metric

The searched hyperparameters include:

depth

learning_rate

l2_leaf_reg

The optimized model is saved in CatBoost's native .cbm format for reliable loading during inference.

Model Performance

The final application is configured around an optimized CatBoost model that achieved:

Metric

Result

Held-out Test Accuracy

94%

Model

CatBoost Classifier

Optimization

RandomizedSearchCV

Cross-Validation

Stratified 5-Fold

The notebook also uses classification reports and confusion matrices to inspect model performance beyond a single accuracy value.

Application Experience

VaultIQ is implemented as an interactive Streamlit application with three main areas:

1. Predict

Users enter applicant and loan information through a guided form. The application then returns:

Loan Approved or Loan Rejected

Approval probability

Gauge visualization

Approval vs. rejection probability chart

Key input ratios and values

Submitted applicant profile

2. Model Insights

The application provides a visual explanation layer using:

Full feature-importance chart

Top-5 driver visualization

Model summary cards

Held-out test accuracy display

3. About

The application documents the project, feature definitions, and feature-type distribution in an accessible interface.

Repository Structure

=======
- **RandomizedSearchCV**
- **Stratified 5-Fold Cross-Validation**
- Accuracy as the primary search metric

The searched hyperparameters include:

- `depth`
- `learning_rate`
- `l2_leaf_reg`

The optimized model is saved in CatBoost's native **`.cbm`** format for reliable loading during inference.

---

## Model Performance

The final application is configured around an optimized CatBoost model that achieved:

| Metric | Result |
|---|---:|
| **Held-out Test Accuracy** | **94%** |
| Model | CatBoost Classifier |
| Optimization | RandomizedSearchCV |
| Cross-Validation | Stratified 5-Fold |

The notebook also uses classification reports and confusion matrices to inspect model performance beyond a single accuracy value.

---

## Application Experience

VaultIQ is implemented as an interactive Streamlit application with three main areas:

### 1. Predict

Users enter applicant and loan information through a guided form. The application then returns:

- **Loan Approved** or **Loan Rejected**
- Approval probability
- Gauge visualization
- Approval vs. rejection probability chart
- Key input ratios and values
- Submitted applicant profile

### 2. Model Insights

The application provides a visual explanation layer using:

- Full feature-importance chart
- Top-5 driver visualization
- Model summary cards
- Held-out test accuracy display

### 3. About

The application documents the project, feature definitions, and feature-type distribution in an accessible interface.

---

## Repository Structure

```text
>>>>>>> 25a3ce7 (Improve The README File)
Loan-Approval-Classification/
│
├── Dataset/
│   ├── loan_data.csv
│   └── catboost_info/
│
├── Loan Approval Application.py      # Streamlit web application
├── Loan Approval Notebook.ipynb      # Full ML workflow and experiments
├── catboost_loan_approval.cbm        # Trained CatBoost model
├── requirements.txt                  # Python dependencies
├── Dockerfile                         # Container configuration
├── .gitignore                         # Git exclusions
└── README.md                          # Project documentation
<<<<<<< HEAD

Run the Project Locally

1. Clone the repository

git clone https://github.com/amir-4/Loan-Approval-Classification.git
cd Loan-Approval-Classification

2. Create and activate a virtual environment

Windows

python -m venv .venv
.\.venv\Scripts\Activate.ps1

macOS / Linux

python3 -m venv .venv
source .venv/bin/activate

3. Install dependencies

python -m pip install --upgrade pip
pip install -r requirements.txt

4. Run the Streamlit application

The current application file is:

Loan Approval Application.py

Run:

python -m streamlit run "Loan Approval Application.py"

Then open the local Streamlit URL shown in the terminal, usually:

http://localhost:8501

Docker

The repository also includes a Dockerfile for containerized deployment.

docker build -t vaultiq .
docker run -p 8501:8501 vaultiq

Note: The current Dockerfile uses app.py as its Streamlit entry point, while the repository's current application file is named Loan Approval Application.py. Before using the Dockerfile as-is, update the Docker entry point to the current filename or rename the application file to app.py.

Deployment
=======
```

---

## Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/amir-4/Loan-Approval-Classification.git
cd Loan-Approval-Classification
```

### 2. Create and activate a virtual environment

#### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Streamlit application

The current application file is:

```text
Loan Approval Application.py
```

Run:

```bash
python -m streamlit run "Loan Approval Application.py"
```

Then open the local Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

---

## Docker

The repository also includes a `Dockerfile` for containerized deployment.

```bash
docker build -t vaultiq .
docker run -p 8501:8501 vaultiq
```

> **Note:** The current Dockerfile uses `app.py` as its Streamlit entry point, while the repository's current application file is named `Loan Approval Application.py`. Before using the Dockerfile as-is, update the Docker entry point to the current filename or rename the application file to `app.py`.

---

## Deployment
>>>>>>> 25a3ce7 (Improve The README File)

The application can be deployed to a service that supports Streamlit or Docker.

Typical options include:

<<<<<<< HEAD
Streamlit Community Cloud

Hugging Face Spaces

Render

Railway

A private VPS or cloud VM using Docker

For a public deployment, make sure the trained .cbm model is available to the application at runtime.

Live Project

Live Demo

Add your deployed application URL here:

[Live Demo](https://loan-approval-classification-bbgqhxe3bryevcs978wsjj.streamlit.app/)

Kaggle Notebook

Add your Kaggle notebook URL here:

[Kaggle Notebook](https://www.kaggle.com/code/amirmansour/loan-approval-classification)

GitHub Repository

View the GitHub Repository

Technology Stack

Area

Technology

Language

Python

Data Processing

pandas, NumPy

Visualization

Matplotlib, Seaborn, Plotly

Machine Learning

scikit-learn

Gradient Boosting

CatBoost, XGBoost, LightGBM

Application

Streamlit

Model Format

CatBoost .cbm

Containerization

Docker

Notebook

Jupyter Notebook

Key Learning Outcomes

This project was built to practice a complete Machine Learning lifecycle, including:

Translating a business problem into a classification task

Working with mixed numerical and categorical data

Building comparable ML baselines

Using cross-validation for robust model selection

Performing hyperparameter optimization

Interpreting feature importance

Saving and loading a trained model

Connecting a trained ML model to a user-facing application

Packaging the project for reproducible local and containerized deployment

Future Improvements

Potential next steps for the project include:

Adding probability calibration and threshold analysis

Expanding model evaluation with ROC-AUC and Precision-Recall curves

Adding automated tests for input validation and inference

Improving explainability with SHAP-based analysis

Adding CI/CD for automated deployment

Monitoring model performance after deployment

Disclaimer

VaultIQ is a portfolio and educational Machine Learning project built for experimentation and demonstration. Its predictions should not be treated as financial advice or as a replacement for professional credit underwriting, regulatory compliance, or human review.

Author

Amir Mansour

GitHub: @amir-4
=======
- Streamlit Community Cloud
- Hugging Face Spaces
- Render
- Railway
- A private VPS or cloud VM using Docker

For a public deployment, make sure the trained `.cbm` model is available to the application at runtime.

---

## Live Project

### Live Demo

**Add your deployed application URL here:**

`[Live Demo](YOUR_LIVE_DEMO_URL)`

### Kaggle Notebook

**Add your Kaggle notebook URL here:**

`[Kaggle Notebook](YOUR_KAGGLE_NOTEBOOK_URL)`

### GitHub Repository

[View the GitHub Repository](https://github.com/amir-4/Loan-Approval-Classification)

---

## Technology Stack

| Area | Technology |
|---|---|
| Language | Python |
| Data Processing | pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | scikit-learn |
| Gradient Boosting | CatBoost, XGBoost, LightGBM |
| Application | Streamlit |
| Model Format | CatBoost `.cbm` |
| Containerization | Docker |
| Notebook | Jupyter Notebook |

---

## Key Learning Outcomes

This project was built to practice a complete Machine Learning lifecycle, including:

- Translating a business problem into a classification task
- Working with mixed numerical and categorical data
- Building comparable ML baselines
- Using cross-validation for robust model selection
- Performing hyperparameter optimization
- Interpreting feature importance
- Saving and loading a trained model
- Connecting a trained ML model to a user-facing application
- Packaging the project for reproducible local and containerized deployment

---

## Future Improvements

Potential next steps for the project include:

- Adding probability calibration and threshold analysis
- Expanding model evaluation with ROC-AUC and Precision-Recall curves
- Adding automated tests for input validation and inference
- Improving explainability with SHAP-based analysis
- Adding CI/CD for automated deployment
- Monitoring model performance after deployment

---

## Disclaimer

VaultIQ is a **portfolio and educational Machine Learning project** built for experimentation and demonstration. Its predictions should not be treated as financial advice or as a replacement for professional credit underwriting, regulatory compliance, or human review.

---

## Author

**Amir Mansour**

GitHub: [@amir-4](https://github.com/amir-4)

---
>>>>>>> 25a3ce7 (Improve The README File)

<p align="center">
  <strong>VaultIQ</strong><br>
  <sub>Turning financial data into intelligent loan predictions.</sub>
</p>
