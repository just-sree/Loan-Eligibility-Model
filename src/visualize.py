import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.logger import get_logger

logger = get_logger(__name__)

def plot_confusion_matrix(conf_matrix):
    try:
        plt.figure(figsize=(8,6))  # figsize specifies the width and height of the figure
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')  # annot adds annotations, cmap sets the color map
        plt.xlabel('Predicted Labels')  # xlabel sets the label for the x-axis
        plt.ylabel('True Labels')  # ylabel sets the label for the y-axis
        plt.title('Confusion Matrix')
        plt.show()
        logger.info("Confusion matrix plotted.")
    except Exception as e:
        logger.exception(f"Visualization failed: {e}")  # Log the exception message
        raise
