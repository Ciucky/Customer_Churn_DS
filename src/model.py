import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

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

def train_model(data):
    logging.info("Starting model training")
    
    # Define features and target
    X = data.drop(columns=['Churn Value'])
    y = data['Churn Value']
    
    # Check target variable distribution
    logging.info("Target variable distribution: %s", y.value_counts())
    
    # Split data into training and testing sets
    logging.info("Splitting data into training and testing sets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Check distribution in training and testing sets
    logging.info("Training set target distribution: %s", y_train.value_counts())
    logging.info("Testing set target distribution: %s", y_test.value_counts())
    
    # Handle class imbalance using SMOTE
    logging.info("Applying SMOTE to handle class imbalance")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Check distribution in resampled training set
    logging.info("Resampled training set target distribution: %s", y_train_resampled.value_counts())
    
    # Train logistic regression model
    logging.info("Training logistic regression model")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_resampled, y_train_resampled)
    
    # Evaluate the model
    logging.info("Evaluating the model")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    logging.info("Classification Report:\n%s", classification_report(y_test, y_pred))
    logging.info("Confusion Matrix:\n%s", confusion_matrix(y_test, y_pred))
    logging.info("ROC AUC Score: %f", roc_auc_score(y_test, y_prob))
    
    # Save the model
    model_path = 'models/logistic_regression_model.pkl'
    joblib.dump(model, model_path)
    logging.info("Model saved to %s", model_path)
    
    return model

if __name__ == "__main__":
    enhanced_data_path = 'data/processed/enhanced_data.csv'
    
    data = load_data(enhanced_data_path)
    model = train_model(data)
