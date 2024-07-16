import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

def eda(data):
    logging.info("Starting Exploratory Data Analysis (EDA)")
    
    # Print column names for verification
    logging.info("Data columns: %s", data.columns)
    #print("Data columns:", data.columns)
    
    # Summary statistics
    logging.info("Generating summary statistics")
    summary = data.describe()
    summary.to_csv('reports/summary_statistics.csv')
    logging.info("Summary statistics saved to reports/summary_statistics.csv")
    
    # Data distribution plots
    if 'Tenure Months' in data.columns:
        logging.info("Generating distribution plot for 'Tenure Months'")
        plt.figure(figsize=(12, 6))
        sns.histplot(data['Tenure Months'], kde=True)
        plt.title('Distribution of Tenure Months')
        plt.savefig('reports/figures/tenure_distribution.png')
        plt.close()
    
    if 'Monthly Charges' in data.columns:
        logging.info("Generating distribution plot for 'Monthly Charges'")
        plt.figure(figsize=(12, 6))
        sns.histplot(data['Monthly Charges'], kde=True)
        plt.title('Distribution of Monthly Charges')
        plt.savefig('reports/figures/monthly_charge_distribution.png')
        plt.close()
    
    if 'Total Charges' in data.columns:
        logging.info("Generating distribution plot for 'Total Charges'")
        plt.figure(figsize=(12, 6))
        sns.histplot(data['Total Charges'], kde=True)
        plt.title('Distribution of Total Charges')
        plt.savefig('reports/figures/total_charge_distribution.png')
        plt.close()
    
    # Correlation heatmap
    logging.info("Generating correlation heatmap")
    plt.figure(figsize=(15, 10))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig('reports/figures/correlation_heatmap.png')
    plt.close()
    logging.info("Correlation heatmap saved")

    # Churn distribution by categorical features
    categorical_features = ['Gender_Male', 'Senior Citizen_Yes', 'Partner_Yes', 'Dependents_Yes', 'Phone Service_Yes',
                            'Multiple Lines_No phone service', 'Multiple Lines_Yes', 'Internet Service_Fiber optic',
                            'Internet Service_No', 'Online Security_No internet service', 'Online Security_Yes',
                            'Online Backup_No internet service', 'Online Backup_Yes', 'Device Protection_No internet service',
                            'Device Protection_Yes', 'Tech Support_No internet service', 'Tech Support_Yes', 
                            'Streaming TV_No internet service', 'Streaming TV_Yes', 'Streaming Movies_No internet service',
                            'Streaming Movies_Yes', 'Contract_One year', 'Contract_Two year', 'Paperless Billing_Yes',
                            'Payment Method_Credit card (automatic)', 'Payment Method_Electronic check', 
                            'Payment Method_Mailed check']

    for feature in categorical_features:
        if feature in data.columns:
            logging.info(f"Generating distribution plot for {feature} by Churn Value")
            plt.figure(figsize=(12, 6))
            sns.countplot(x=feature, hue='Churn Value', data=data)
            plt.title(f'Distribution of {feature} by Churn Value')
            plt.savefig(f'reports/figures/{feature}_churn_distribution.png')
            plt.close()
    
    logging.info("EDA completed")

if __name__ == "__main__":
    processed_data_path = 'data/processed/processed_data.csv'
    
    data = load_data(processed_data_path)
    eda(data)
