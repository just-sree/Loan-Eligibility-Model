import pandas as pd
import os

def load_dataset(file_name: str):
    """
    Load the dataset from the specified file.

    Args:
        file_name (str): The name of the dataset file.

    Returns:
        pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file '{file_name}' not found at '{file_path}'")
    return pd.read_csv(file_path)