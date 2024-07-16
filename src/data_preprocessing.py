import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(filename='logs/project.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s - %(message)s')

def load_data(file_path):
    logging.info("Loading data from %s", file_path)
    try:
        data = pd.read_excel(file_path)
        logging.info("Data loaded successfully")
        return data
    except Exception as e:
        logging.error("Error loading data: %s", e)
        raise

def preprocess_data(data):
    logging.info("Starting data preprocessing")
    
    # Drop columns not needed for analysis
    columns_to_drop = ['CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code', 'Lat Long']
    data.drop(columns=columns_to_drop, inplace=True)
    
    # Convert Total Charges to numeric, coerce errors to NaN
    data['Total Charges'] = pd.to_numeric(data['Total Charges'], errors='coerce')
    
    # Fill missing values for non-churn customers
    data.loc[data['Churn Value'] == 0, 'Churn Reason'] = data.loc[data['Churn Value'] == 0, 'Churn Reason'].fillna('No churn')
    
    # Now fill other missing values
    missing_values_before = data.isnull().sum().sum()
    data.fillna(data.median(numeric_only=True), inplace=True)
    missing_values_after = data.isnull().sum().sum()
    logging.info("Handled missing values: before = %d, after = %d", missing_values_before, missing_values_after)
    
    # Encode categorical variables
    categorical_cols = ['Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Phone Service', 'Multiple Lines',
                        'Internet Service', 'Online Security', 'Online Backup', 'Device Protection', 'Tech Support',
                        'Streaming TV', 'Streaming Movies', 'Contract', 'Paperless Billing', 'Payment Method',
                        'Churn Label', 'Churn Reason']
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
    logging.info("Data preprocessing completed")
    return data

if __name__ == "__main__":
    raw_data_path = 'data/raw/Telco_customer_churn.xlsx'
    processed_data_path = 'data/processed/processed_data.csv'
    
    data = load_data(raw_data_path)
    processed_data = preprocess_data(data)
    processed_data.to_csv(processed_data_path, index=False)
    logging.info("Processed data saved to %s", processed_data_path)
