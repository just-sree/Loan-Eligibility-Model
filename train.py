import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data_preprocessing import load_and_process_data
from src.train_model import train_and_save_model
from src.evaluate_model import evaluate_model
from src.visualize import plot_confusion_matrix

def main():
    data_path = 'data/credit.csv'
    model_path = 'models/loan_model.pkl'

    X_train, X_test, y_train, y_test = load_and_process_data(data_path)
    model = train_and_save_model(X_train, y_train, model_path)
    eval_results = evaluate_model(model, X_test, y_test)

    print("Accuracy:", eval_results["accuracy"])
    print(eval_results["classification_report"])

    plot_confusion_matrix(eval_results["confusion_matrix"])

if __name__ == "__main__":
    main()
