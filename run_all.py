import subprocess
import logging

# Configure logging
logging.basicConfig(filename='logs/run_all.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_script(script_path):
    try:
        logging.info("Running script: %s", script_path)
        subprocess.run(["python", script_path], check=True)
        logging.info("Successfully ran script: %s", script_path)
    except subprocess.CalledProcessError as e:
        logging.error("Error running script: %s", script_path)
        logging.error(e)
        raise

if __name__ == "__main__":
    scripts = [
        "src/data_preprocessing.py",
        "src/exploratory_data_analysis.py",
        "src/feature_engineering.py",
        "src/model.py",
        "src/evaluate.py",
        "src/gen_report.py"
    ]
    
    for script in scripts:
        run_script(script)
