# PowerShell Script: setup_loan_project.ps1

# Set up main project directory
$project = "D:\Personal Projects\Loan-Eligibility-Model"
New-Item -ItemType Directory -Path $project -Force | Out-Null
Set-Location $project

# Create subdirectories
New-Item -ItemType Directory -Path "data" | Out-Null
New-Item -ItemType Directory -Path "models" | Out-Null
New-Item -ItemType Directory -Path "src" | Out-Null

# Create __init__.py in src/
New-Item -ItemType File -Path "src\__init__.py" -Force | Out-Null

# Create logger.py
$loggerCode = @'
import logging

def get_logger(name=__name__):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
'@
Set-Content -Path "src\logger.py" -Value $loggerCode

# Populate src/data_preprocessing.py
$dataPreprocessing = @'
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from .logger import get_logger

logger = get_logger(__name__)

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Data loaded from {filepath} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def preprocess_data(df, scale_numeric=False):
    try:
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)

        if 'Loan_Approved' in df.columns:
            df['Loan_Approved'] = df['Loan_Approved'].map({'Y': 1, 'N': 0})

        cat_cols = df.select_dtypes(include='object').columns.tolist()
        if 'Loan_Approved' in cat_cols:
            cat_cols.remove('Loan_Approved')

        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        if scale_numeric:
            scaler = StandardScaler()
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Loan_Approved' in num_cols:
                num_cols.remove('Loan_Approved')
            df[num_cols] = scaler.fit_transform(df[num_cols])

        logger.info("Preprocessing complete")
        return df
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        return df
'@
Set-Content -Path "src\data_preprocessing.py" -Value $dataPreprocessing

# Populate src/train_models.py
$trainModels = @'
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
'@
Set-Content -Path "src\train_models.py" -Value $trainModels

# Populate src/evaluate.py
$evaluateCode = @'
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
'@
Set-Content -Path "src\evaluate.py" -Value $evaluateCode

# Populate src/visualization.py
$visualizationCode = @'
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from .logger import get_logger

logger = get_logger(__name__)

def plot_target_distribution(df):
    try:
        sns.countplot(x='Loan_Approved', data=df)
        plt.title("Loan Approval Distribution")
        plt.show()
    except Exception as e:
        logger.error(f"Target distribution plot error: {e}")

def plot_confusion_matrix(y_true, y_pred):
    try:
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
    except Exception as e:
        logger.error(f"Confusion matrix plot error: {e}")
'@
Set-Content -Path "src\visualization.py" -Value $visualizationCode

# Create placeholder files
"# Streamlit App Entry Point" | Set-Content -Path "app.py"
"# Model Training Script" | Set-Content -Path "train.py"

# Create requirements.txt
$requirements = @'
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
joblib
'@
Set-Content -Path "requirements.txt" -Value $requirements

# Create README.md
$readme = @'
# 🏦 Loan Approval Prediction

This project uses Logistic Regression and Random Forest (with hyperparameter tuning) to predict whether a loan should be approved based on applicant data.

## 💠 How to Use

1. Place your dataset as `loan_data.csv` in the `data/` folder.

2. Train the models:
   ```
   python train.py
   ```

3. Launch the Streamlit app:
   ```
   streamlit run app.py
   ```

## 📂 Folder Structure

- `data/`: Contains your dataset.
- `models/`: Trained models are saved here.
- `src/`: All modular code (preprocessing, training, evaluation, logging).
- `app.py`: Streamlit app entry point.
- `train.py`: Model training script.

## ✅ Requirements
Install dependencies with:
```
pip install -r requirements.txt
```
'@
Set-Content -Path "README.md" -Value $readme

# Create placeholder dataset
New-Item -ItemType File -Path "data\loan_data.csv" -Force | Out-Null

Write-Host "`n✅ Full Loan Approval Project scaffold with script content created!"
Write-Host "Project directory: $project"
