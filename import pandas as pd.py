import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('data.csv')
df.head()
df.describe()
print("Current columns:", df.columns.tolist())
df.isnull().sum()
duplicates = df.duplicated().sum()
print(duplicates)
# In your code/report:
print("Data types:")
print(df.dtypes)
print("Unique values per column:")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} values")
print("Current gender values:")
print(df['gender'].value_counts())
print(df['work_environment'].value_counts())
print(df['employment_status'].value_counts())
print(df['mental_health_history'].value_counts())
print(df['seeks_treatment'].value_counts())
print(df['mental_health_risk'].value_counts())
categorical_cols = ['gender', 'work_environment', 'employment_status', 
                    'mental_health_history', 'mental_health_risk']

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
# Example 1: Create mental health composite score
df['mental_health_composite'] = (
    df['depression_score'] + df['anxiety_score'] - df['social_support_score']
)

# Example 2: Categorize stress levels
df['stress_category'] = pd.cut(
    df['stress_level'], 
    bins=[0, 3, 7, 10],
    labels=['Low', 'Medium', 'High']
)
# Example 3: Calculate sleep efficiency ratio
df['sleep_activity_ratio'] = df['sleep_hours'] / (df['physical_activity_days'] + 1)
# Identify outliers using IQR
Q1 = df[['stress_level', 'depression_score']].quantile(0.25)
Q3 = df[['stress_level', 'depression_score']].quantile(0.75)
IQR = Q3 - Q1

# Decide: Cap, remove, or keep with explanation
# Option: Cap outliers
df['stress_level_capped'] = np.where(
    df['stress_level'] > (Q3['stress_level'] + 1.5 * IQR['stress_level']),
    Q3['stress_level'] + 1.5 * IQR['stress_level'],
    df['stress_level']
)
# One-hot encode stress_category (you created it)
df = pd.get_dummies(df, columns=['stress_category'], drop_first=True)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['seeks_treatment_encoded'] = le.fit_transform(df['seeks_treatment'])
X = df.drop(['seeks_treatment', 'seeks_treatment_encoded'], axis=1)
y = df['seeks_treatment_encoded']
print(f"Final feature shape: {X.shape}")
print(f"Target distribution: {y.value_counts(normalize=True)}")
df.to_csv('data_preprocessed.csv', index=False)


# Select numerical columns for correlation
numerical_cols = ['age', 'stress_level', 'sleep_hours', 'physical_activity_days',
                  'depression_score', 'anxiety_score', 'social_support_score',
                  'productivity_score', 'mental_health_composite', 'sleep_activity_ratio']

corr_matrix = df[numerical_cols].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='BuPu', center=0, fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300)
plt.show()

# Identify top correlations with target
target_corr = df[numerical_cols + ['seeks_treatment_encoded']].corr()['seeks_treatment_encoded'].abs().sort_values(ascending=False)
print("\\nTop correlations with seeks_treatment:")
print(target_corr.head(10))
 #2. T-test for Treatment Seekers vs Non-Seekers

from scipy import stats

# Compare means between groups
group_yes = df[df['seeks_treatment'] == 'Yes']
group_no = df[df['seeks_treatment'] == 'No']

print("=== T-TEST RESULTS: Treatment Seekers vs Non-Seekers ===\\n")

for col in ['depression_score', 'anxiety_score', 'stress_level', 'social_support_score']:
    t_stat, p_val = stats.ttest_ind(group_yes[col], group_no[col])
    print(f"{col}:")
    print(f"  Treatment Yes (n={len(group_yes)}): Mean = {group_yes[col].mean():.2f}")
    print(f"  Treatment No  (n={len(group_no)}): Mean = {group_no[col].mean():.2f}")
    print(f"  t-statistic = {t_stat:.3f}, p-value = {p_val:.4f}")
    print(f"  Interpretation: {'Significant difference' if p_val < 0.05 else 'No significant difference'}\\n")
# A) Distribution of target
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
df['seeks_treatment'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Target Distribution: Seeks Treatment')
plt.xlabel('Seeks Treatment')
plt.ylabel('Count')
plt.xticks(rotation=0)

# B) Boxplot: Depression score by treatment
plt.subplot(1, 2, 2)
sns.boxplot(x='seeks_treatment', y='depression_score', data=df, palette=['skyblue', 'salmon'])
plt.title('Depression Score by Treatment Seeking')
plt.xlabel('Seeks Treatment')
plt.ylabel('Depression Score')

plt.tight_layout()
plt.savefig('target_analysis.png', dpi=300)
plt.show()

# C) Scatter: Depression vs Anxiety colored by treatment
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='depression_score', y='anxiety_score', 
                hue='seeks_treatment', alpha=0.6, palette=['skyblue', 'salmon'])
plt.title('Depression vs Anxiety Scores (Colored by Treatment Seeking)')
plt.xlabel('Depression Score')
plt.ylabel('Anxiety Score')
plt.legend(title='Seeks Treatment')
plt.grid(True, alpha=0.3)
plt.savefig('depression_anxiety_scatter.png', dpi=300)
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assuming df is your DataFrame
# Convert target to numeric
df['seeks_treatment'] = df['seeks_treatment'].map({'Yes': 1, 'No': 0})

# Final split
X = df.drop(columns=['seeks_treatment'])
y = df['seeks_treatment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numerical features only
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

from sklearn.model_selection import cross_val_score, GridSearchCV

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Hyperparameter tuning (optional but good for marks)
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2', None],
    'solver': ['lbfgs', 'liblinear']
}

grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42),
                    param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
print(f"Best CV accuracy: {grid.best_score_:.4f}")

from sklearn.metrics import roc_curve, ConfusionMatrixDisplay

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_score(y_test, y_pred_prob):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
plt.savefig('roc_curve.png', dpi=300)

# Confusion Matrix Heatmap
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
plt.savefig('confusion_matrix.png', dpi=300)

# Feature Importance Plot
top_features = feature_importance.head(10)
plt.figure(figsize=(10, 6))
colors = ['green' if coef > 0 else 'red' for coef in top_features['Coefficient']]
plt.barh(top_features['Feature'], top_features['Abs_Importance'], color=colors)
plt.xlabel('Absolute Coefficient Value')
plt.title('Top 10 Most Important Features')
plt.gca().invert_yaxis()
plt.show()
plt.savefig('feature_importance.png', dpi=300)
