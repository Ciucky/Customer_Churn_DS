import pandas as pd
import logging
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

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

def load_model(model_path):
    logging.info("Loading model from %s", model_path)
    try:
        model = joblib.load(model_path)
        logging.info("Model loaded successfully from %s", model_path)
        return model
    except Exception as e:
        logging.error("Error loading model from %s: %s", model_path, e)
        raise

def evaluate_model(model, X_test, y_test):
    logging.info("Starting model evaluation")
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    logging.info("Generating classification report")
    class_report = classification_report(y_test, y_pred)
    logging.info("Classification Report:\n%s", class_report)
    
    logging.info("Generating confusion matrix")
    conf_matrix = confusion_matrix(y_test, y_pred)
    logging.info("Confusion Matrix:\n%s", conf_matrix)
    
    logging.info("Calculating ROC AUC score")
    roc_auc = roc_auc_score(y_test, y_prob)
    logging.info("ROC AUC Score: %f", roc_auc)
    
    logging.info("Calculating accuracy score")
    accuracy = accuracy_score(y_test, y_pred)
    logging.info("Accuracy Score: %f", accuracy)
    
    # Save classification report
    with open('reports/classification_report.txt', 'w') as f:
        f.write(class_report)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('reports/figures/confusion_matrix.png')
    plt.close()
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.savefig('reports/figures/roc_curve.png')
    plt.close()
    
    logging.info("Model evaluation completed")

if __name__ == "__main__":
    test_data_path = 'data/processed/enhanced_data.csv' # Assuming the enhanced data contains the test set
    model_path = 'models/logistic_regression_model.pkl'
    
    # Load data and model
    data = load_data(test_data_path)
    model = load_model(model_path)
    
    # Define features and target
    X = data.drop(columns=['Churn Value'])
    y = data['Churn Value']
    
    # Evaluate the model
    evaluate_model(model, X, y)
