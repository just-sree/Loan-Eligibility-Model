import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.linear_model import LogisticRegression
import joblib
from src.logger import get_logger

logger = get_logger(__name__)

def train_and_save_model(X_train, y_train, model_path='models/loan_model.pkl'):
    try:
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        logger.info(f"Model trained and saved to {model_path}")
        return model
    except Exception as e:
        logger.exception("Model training failed.")
        raise