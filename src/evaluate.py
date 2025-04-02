from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from .logger import get_logger

logger = get_logger(__name__)

def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        logger.info(f"Accuracy: {acc}")
        logger.info(f"Confusion Matrix:\n{cm}")
        logger.info(f"Classification Report:\n{report}")

        return acc, cm, report
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        return None, None, None
