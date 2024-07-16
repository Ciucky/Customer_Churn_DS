# Customer Churn Prediction Project

## Project Overview
This project aims to analyse and predict customer churn for a telecom company using the Telco Customer Churn dataset. The primary goal is to identify the factors that contribute to customer churn and develop a predictive model to forecast which customers are likely to leave the company. The insights gained from this analysis can help the company implement targeted retention strategies.

## Project Structure
```
customer_churn_analysis/
│
├── data/
│ ├── raw/
│ │ └── Telco_customer_churn.xlsx
│ ├── processed/
│ │ ├── processed_data.csv
│ │ └── enhanced_data.csv
│
├── src/
│ ├── data_preprocessing.py
│ ├── exploratory_data_analysis.py
│ ├── feature_engineering.py
│ ├── model.py
│ ├── evaluate.py
│ └── gen_report.py
│
├── reports/
│ ├── figures/
│ │ ├── confusion_matrix.png
│ │ ├── roc_curve.png
│ │ ├── lots of other graphs
│ └── generated_report.pdf
│ └── classification_report.txt
│ └── summary_statistics.csv
│
├── logs/
│ ├── project.log
│ └── run_evaluate_report.log
│
├── README.md
├── requirements.txt
├── run_all.py
└── run_evaluate_report.py
```

## Scripts and Their Functions

1. **data_preprocessing.py**:
    - Loads the raw data, preprocesses it by handling missing values and encoding categorical variables, and saves the processed data.

2. **exploratory_data_analysis.py**:
    - Performs exploratory data analysis (EDA) on the processed data, generates visualizations, and saves them to the reports directory.

3. **feature_engineering.py**:
    - Engineers new features from the processed data and saves the enhanced dataset.

4. **model.py**:
    - Splits the enhanced data into training and testing sets, handles class imbalance, trains a logistic regression model, evaluates the model, and saves the trained model.

5. **evaluate.py**:
    - Loads the trained model and test data, evaluates the model performance, generates relevant metrics and visualizations, and saves them to the reports directory.

6. **gen_report.py**:
    - Compiles the results of the model evaluation into a comprehensive PDF report.

7. **run_all.py**:
    - Runs all the scripts in the correct sequence: data preprocessing, feature engineering, model building, model evaluation, and report generation.

8. **run_evaluate_report.py**:
    - Runs only the model evaluation and report generation scripts.

## How to Run the Scripts

1. **Install Dependencies**:
    - Ensure you have Python installed. You can create a virtual environment and install the required dependencies using the following commands:
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

2. **Run All Scripts**:
    - To run the entire project from data preprocessing to report generation, execute:
    ```bash
    python run_all.py
    ```

3. **Run Model Evaluation and Report Generation Only**:
    - To run only the model evaluation and report generation scripts, execute:
    ```bash
    python run_evaluate_report.py
    ```

## Dependencies

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- fpdf
- joblib

You can install these dependencies using the provided `requirements.txt` file.

## Data Description

The dataset contains 7043 observations with 33 variables (features), including customer demographic information, services subscribed to, account information, and churn details. Key features include:

- **Customer Demographics**: Gender, Senior Citizen, Partner, Dependents
- **Service Information**: Phone Service, Multiple Lines, Internet Service, Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies
- **Account Information**: Contract, Paperless Billing, Payment Method, Monthly Charges, Total Charges, Tenure Months
- **Churn Details**: Churn Label, Churn Value, Churn Score, CLTV, Churn Reason

## Objectives

1. **Data Preprocessing**: Clean and preprocess the data, handle missing values, and encode categorical variables.
2. **Exploratory Data Analysis (EDA)**: Understand the data distribution and identify patterns and correlations.
3. **Feature Engineering**: Create new features to improve model performance.
4. **Model Building**: Develop a predictive model to forecast customer churn using various machine learning algorithms.
5. **Model Evaluation**: Assess the model's performance using appropriate metrics and validate its predictive power.
6. **Reporting and Visualization**: Generate insightful visualizations and a comprehensive report to communicate findings and recommendations.

## Documentation and Logging

- Each script logs its progress and any issues encountered in the `logs` directory.
- The final report and visualizations are saved in the `reports` directory.
