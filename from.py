from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv('data_preprocessed.csv')

# Final split
X = data.drop(columns=['seeks_treatment'])
y = data['seeks_treatment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numerical features only (not the one-hot encoded booleans)
numerical_cols = ['age', 'stress_level', 'sleep_hours', 'physical_activity_days',
                  'depression_score', 'anxiety_score', 'social_support_score',
                  'productivity_score', 'mental_health_composite', 'sleep_activity_ratio',
                  'stress_level_capped']

scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Target balance: {np.mean(y_train):.3f} positive")


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# Quick evaluation
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Full evaluation
y_pred_prob = model.predict_proba(X_test)[:, 1]

print("=== MODEL EVALUATION ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_prob):.4f}")
print("\\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))

# Get coefficients
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': model.coef_[0]
})
feature_importance['Abs_Importance'] = np.abs(feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values('Abs_Importance', ascending=False)

print("\\n=== TOP 10 MOST IMPORTANT FEATURES ===")
for idx, row in feature_importance.head(10).iterrows():
    direction = "INCREASES" if row['Coefficient'] > 0 else "DECREASES"
    print(f"{row['Feature']:30} → {direction} treatment seeking (coef: {row['Coefficient']:.4f})")

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Feature Importance Plot
plt.figure(figsize=(10, 6))
top_10 = feature_importance.head(10)
colors = ['red' if coef < 0 else 'green' for coef in top_10['Coefficient']]
plt.barh(top_10['Feature'], top_10['Coefficient'], color=colors)
plt.xlabel('Coefficient Value')
plt.title('Top 10 Feature Importance in Predicting Treatment Seeking')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)

# 2. ROC Curve
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_pred_prob):.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.savefig('roc_curve.png', dpi=300)
plt.show()