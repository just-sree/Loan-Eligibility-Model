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
