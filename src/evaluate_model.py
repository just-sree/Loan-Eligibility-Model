import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.logger import get_logger

logger = get_logger(__name__)

def evaluate_model(model, X_test, y_test):
    try:
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)
        report = classification_report(y_test, predictions)

        logger.info(f"Evaluation completed. Accuracy: {accuracy}")

        return {
            "accuracy": accuracy,
            "confusion_matrix": conf_matrix,
            "classification_report": report
        }
    except Exception as e:
        logger.exception("Error during model evaluation.")
        raise
