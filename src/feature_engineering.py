import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(filename='logs/project.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s - %(message)s')

def load_data(file_path):
    logging.info("Loading data from %s", file_path)
    try:
        data = pd.read_csv(file_path)
        logging.info("Data loaded successfully from %s", file_path)
        return data
    except Exception as e:
        logging.error("Error loading data from %s: %s", file_path, e)
        raise

def engineer_features(data):
    logging.info("Starting feature engineering")

    # Example Feature Engineering: 
    # 1. Creating interaction terms
    logging.info("Creating interaction terms")
    data['Monthly*Tenure'] = data['Monthly Charges'] * data['Tenure Months']
    
    # 2. Aggregating binary columns to create new features
    logging.info("Aggregating binary columns to create new features")
    data['Total_Services'] = (data[['Phone Service_Yes', 'Multiple Lines_Yes', 'Internet Service_Fiber optic',
                                    'Internet Service_No', 'Online Security_Yes', 'Online Backup_Yes',
                                    'Device Protection_Yes', 'Tech Support_Yes', 'Streaming TV_Yes',
                                    'Streaming Movies_Yes']].sum(axis=1))
    
    # 3. Normalizing numerical features
    logging.info("Normalizing numerical features")
    data['Monthly Charges'] = (data['Monthly Charges'] - data['Monthly Charges'].mean()) / data['Monthly Charges'].std()
    data['Total Charges'] = (data['Total Charges'] - data['Total Charges'].mean()) / data['Total Charges'].std()
    data['Tenure Months'] = (data['Tenure Months'] - data['Tenure Months'].mean()) / data['Tenure Months'].std()
    
    logging.info("Feature engineering completed")
    return data

if __name__ == "__main__":
    processed_data_path = 'data/processed/processed_data.csv'
    enhanced_data_path = 'data/processed/enhanced_data.csv'
    
    data = load_data(processed_data_path)
    enhanced_data = engineer_features(data)
    enhanced_data.to_csv(enhanced_data_path, index=False)
    logging.info("Enhanced data saved to %s", enhanced_data_path)
