import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import get_logger

logger = get_logger(__name__)

def load_and_process_data(filepath):
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")

        if 'Loan_ID' in df.columns:
            df.drop(columns=['Loan_ID'], inplace=True)

        if 'Loan_Approved' not in df.columns:
            raise ValueError("Target column 'Loan_Approved' not found.")

        df['Loan_Approved'] = df['Loan_Approved'].map({'Y': 1, 'N': 0})

        # Handle missing values
        fill_mode = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History']
        for col in fill_mode:
            df[col].fillna(df[col].mode()[0], inplace=True)

        df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)

        logger.info("Missing values filled.")

        df = pd.get_dummies(df, drop_first=True)
        logger.info("Categorical variables encoded.")

        X = df.drop('Loan_Approved', axis=1)
        y = df['Loan_Approved']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        logger.info("Train-test split done.")

        return X_train, X_test, y_train, y_test

    except Exception as e:
        logger.exception(f"Data preprocessing failed: {e}")
        raise
