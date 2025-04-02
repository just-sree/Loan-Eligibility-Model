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
