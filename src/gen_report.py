import pandas as pd
import logging
from fpdf import FPDF

# Configure logging
logging.basicConfig(filename='logs/project.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s - %(message)s')

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Customer Churn Prediction Report', 0, 1, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

    def add_image(self, image_path, x=None, y=None, w=0, h=0):
        self.image(image_path, x, y, w, h)
        self.ln(10)

def create_report():
    logging.info("Creating PDF report")
    pdf = PDFReport()
    pdf.add_page()

    # Add classification report
    try:
        with open('reports/classification_report.txt', 'r') as file:
            class_report = file.read()
        pdf.chapter_title('Classification Report')
        pdf.chapter_body(class_report)
    except Exception as e:
        logging.error("Error reading classification report: %s", e)

    # Add confusion matrix image
    try:
        pdf.chapter_title('Confusion Matrix')
        pdf.add_image('reports/figures/confusion_matrix.png', w=150)
    except Exception as e:
        logging.error("Error adding confusion matrix image: %s", e)

    # Add ROC curve image
    try:
        pdf.chapter_title('ROC Curve')
        pdf.add_image('reports/figures/roc_curve.png', w=150)
    except Exception as e:
        logging.error("Error adding ROC curve image: %s", e)

    # Save the PDF
    report_path = 'reports/generated_report.pdf'
    pdf.output(report_path)
    logging.info("PDF report saved to %s", report_path)

if __name__ == "__main__":
    create_report()
