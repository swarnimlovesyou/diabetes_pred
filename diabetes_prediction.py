import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load data (downloaded from https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset.)
df = pd.read_csv('diabetes_data_upload.csv')  

# Encoding in the main data frame
# List of categorical columns to encode
categorical_columns = ['Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia', 
                       'Genital thrush', 'visual blurring', 'Itching', 'Irritability', 'delayed healing', 
                       'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity', 'class']

# Initializing the LabelEncoder
le = LabelEncoder()

# Encoding the categorical columns
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Defining features (X) and target (y)
X = df.drop(columns=['class'])
y = df['class']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
xgb_model = xgb.XGBClassifier(random_state=42)
dt = DecisionTreeClassifier(random_state=42)

# Dictionary to store models
models = {
    'MLP': mlp,
    'XGBoost': xgb_model,
    'Decision Tree': dt
}

# Train and evaluate models
results = {}
plt.figure(figsize=(10, 8))

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(classification_report(y_test, y_pred))
    
    results[name] = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

# Finalize ROC plot
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Models')
plt.legend()
plt.show()

# Plot confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Confusion Matrices')

for (name, result), ax in zip(results.items(), axes):
    cm = confusion_matrix(y_test, result['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    ax.set_title(name)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.show()

# Print metrics
print("\nModel Performance Metrics:")
for name, result in results.items():
    print(f"\n{name}:")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"ROC AUC: {result['roc_auc']:.4f}")