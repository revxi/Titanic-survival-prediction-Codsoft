# Titanic Survival Prediction 🚢

This project uses the **Titanic dataset** to predict whether a passenger survived the Titanic disaster using a machine learning model.

## 📊 Dataset

Dataset source: [Kaggle - Titanic Dataset by yasserh](https://www.kaggle.com/datasets/yasserh/titanic-dataset)  
Make sure to download and place `Titanic-Dataset.csv` in the same folder as the script.

## 💡 Features Used
- Age
- Sex
- Pclass (ticket class)
- Fare
- Embarked
- Cabin (encoded)
- SibSp, Parch

## 🔧 Technologies Used
- Python
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn (Random Forest Classifier)

## 🧠 Steps Performed
1. Data Cleaning
2. Encoding Categorical Features
3. Feature Selection
4. Model Building (Random Forest)
5. Evaluation (Accuracy + Classification Report)
6. Feature Importance Visualization

## 🚀 How to Run

```bash
# 1. Clone the repo or download ZIP
# 2. Navigate to the folder
# 3. Install required packages
pip install -r requirements.txt

# 4. Run the script
python titanic_survival_prediction.py
```

## 📈 Sample Output

- Model Accuracy
- Classification Report (Precision, Recall, F1-score)
- Feature importance bar chart

## 📌 Note
- This is a beginner-friendly project. You can try improving it using other models like Logistic Regression, SVM, or XGBoost.
