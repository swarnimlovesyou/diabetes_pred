# README for Diabetes Detection Using Machine Learning Models

## Overview
This repository contains code to classify and predict the risk of diabetes using three different machine learning models:
- Multilayer Perceptron (MLP)
- XGBoost Classifier
- Decision Tree Classifier

The models are evaluated on an early-stage diabetes dataset from the UCI Machine Learning Repository.

## Dataset
The dataset used is the "Early Stage Diabetes Risk Prediction Dataset," which can be downloaded from the UCI Machine Learning Repository:
[Dataset Link](https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset)

The dataset contains features such as:
- Gender
- Polyuria
- Polydipsia
- Sudden weight loss
- Weakness
- Other clinical observations

### Target Variable
- **`class`**: Indicates whether the person has diabetes (`1`) or not (`0`).

## Installation
### Prerequisites
Ensure you have the following Python libraries installed:
- pandas
- numpy
- scikit-learn
- xgboost
- seaborn
- matplotlib

Install them using pip if not already installed:
```bash
pip install pandas numpy scikit-learn xgboost seaborn matplotlib
```

## Code Structure
### Main Steps
1. **Data Loading and Preprocessing:**
   - The dataset is loaded using pandas, and categorical variables are encoded using `LabelEncoder`.

2. **Feature Selection:**
   - Features (`X`) and target (`y`) are defined.

3. **Data Splitting:**
   - The dataset is split into training and testing sets (80% training, 20% testing).

4. **Model Training and Evaluation:**
   - Three machine learning models are trained:
     - Multilayer Perceptron (MLPClassifier)
     - XGBoost (XGBClassifier)
     - Decision Tree (DecisionTreeClassifier)
   - Models are evaluated using:
     - Accuracy
     - ROC AUC Score
     - Classification Report
     - Confusion Matrix

5. **Visualization:**
   - ROC Curves
   - Confusion Matrices

## How to Run the Code
1. Clone the repository:
```bash
git clone <repository-url>
```
2. Navigate to the directory and ensure the dataset is available as `diabetes_data_upload.csv`.
3. Run the script:
```bash
python diabetes_prediction.py
```

## Outputs
1. **Classification Reports:**
   - Includes precision, recall, F1-score, and support for each model.

2. **Performance Metrics:**
   - Accuracy and ROC AUC scores for all models.

3. **Visualizations:**
   - ROC Curves comparing models.
   - Confusion Matrices for all models.

### Example Output
- **Accuracy**
  - MLP: 0.85
  - XGBoost: 0.88
  - Decision Tree: 0.82

- **ROC AUC**
  - MLP: 0.91
  - XGBoost: 0.94
  - Decision Tree: 0.87

## Notes
- Hyperparameters for each model (e.g., hidden layer sizes, max iterations) can be adjusted for better performance.
- Ensure the dataset is preprocessed correctly, with no missing values or incorrect encodings.

## References
1. UCI Machine Learning Repository: Early Stage Diabetes Risk Prediction Dataset
2. [Scikit-learn Documentation](https://scikit-learn.org/)
3. [XGBoost Documentation](https://xgboost.readthedocs.io/)

## License
This project is open-source and available under the MIT License. Feel free to use and modify the code.

