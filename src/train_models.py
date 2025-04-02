import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from .logger import get_logger

logger = get_logger(__name__)

def train_logistic_regression(X_train, y_train):
    try:
        model = LogisticRegression(solver='liblinear')
        model.fit(X_train, y_train)
        logger.info("Logistic Regression trained")
        return model
    except Exception as e:
        logger.error(f"Logistic Regression training error: {e}")
        raise e

def train_random_forest(X_train, y_train):
    try:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 4]
        }
        rf = RandomForestClassifier(random_state=42)
        grid = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)
        logger.info(f"Random Forest best params: {grid.best_params_}")
        return grid.best_estimator_
    except Exception as e:
        logger.error(f"Random Forest training error: {e}")
        raise e

def save_model(model, path):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)
        logger.info(f"Model saved to {path}")
    except Exception as e:
        logger.error(f"Model saving error: {e}")
