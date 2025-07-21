# 💳 Credit Risk Predictor

A machine learning project that predicts whether a loan applicant is likely to default based on their credit and financial data. This classification model helps financial institutions assess credit risk with better accuracy.

---

## 📁 Files in This Repo

- `Main_training_.ipynb` → The main Jupyter notebook with data loading, preprocessing, training, evaluation, and final predictions.
- `train.csv` → The training dataset used to build the model.
- `test.csv` → The test dataset used for final model predictions.
- `README.md` → This file you’re reading.

---

## 📊 Dataset Overview

The datasets contain anonymized information about loan applicants:
- Income
- Credit history
- Loan amount & term
- Employment status
- Debt-to-income ratio
- And more...

🔍 Size: ~31MB `train.csv` ~15MB `test.csv`
📄 Format: CSV

---

## 🧠 Models Used

- Logistic Regression
- Decision Tree
- Random Forest (with both lightweight & full configs)

⚠️ *This notebook is best viewed in segments due to resource constraints on some systems.*

```python
# Resource-efficient config
model = RandomForestClassifier(n_estimators=10, max_depth=5, max_features='sqrt', random_state=42)

# Full-power version (commented out in notebook to prevent crashes)
# model = RandomForestClassifier(n_estimators=100, max_depth=None, max_features='sqrt', random_state=42)
```

---

## 📈 Evaluation Metrics

- Accuracy Score
- Confusion Matrix
- Classification Report
- Feature Importance (Visualized)

---

## 🛠️ Tech Stack

- Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
- Jupyter Notebook
- GitHub for version control

---

## 🔮 Final Predictions on Test Data

After training, the model was used to predict on a separate test dataset (`test.csv`).

---

## 🚀 Future Work

- Add model explainability (SHAP / LIME)
- Try boosting models (XGBoost / LightGBM)
- Deploy as a Streamlit or Flask web app

---

## 📌 How to Run

1. Clone this repo
2. Open `Main_training_.ipynb` in Jupyter
3. Run the cells step by step

> Note: Running heavy models locally may cause Jupyter to crash. You can run the notebook on **Google Colab** or **Kaggle** for better performance.

---

## 🤝 Contributing

Pull requests are welcome! If you spot issues or want to add something cool, feel free to fork and PR 🙌
