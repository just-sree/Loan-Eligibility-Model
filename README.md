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
