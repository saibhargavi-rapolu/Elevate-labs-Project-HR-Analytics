"""
HR Attrition Analysis - (with auto folder creation)
-----------------------------------------------------------------
- Cleans IBM HR Analytics dataset
- Performs EDA
- Builds Logistic Regression model with scaling
- Exports Power BI–ready dataset
"""

# 1. Imports
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import shap

# 2. Load dataset
df = pd.read_csv("C:/my_drive/Elevate_Labs/project/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# 3. Data cleaning
df.drop(['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'], axis=1, inplace=True)

# Encode categorical variables (except AgeGroup which we'll create later)
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# Create Age Groups for Power BI slicers
df['AgeGroup'] = pd.cut(df['Age'], bins=[18, 29, 39, 49, 60],
                        labels=['20-29', '30-39', '40-49', '50-60'])

# 4. EDA

## Attrition count
sns.countplot(x='Attrition', data=df)
plt.title("Attrition Count")
plt.show()

## Attrition by Department
sns.countplot(x='Department', hue='Attrition', data=df)
plt.title("Attrition by Department")
plt.show()

## Correlation heatmap (numeric only)
numeric_df = df.select_dtypes(include=['number'])
plt.figure(figsize=(12,8))
sns.heatmap(numeric_df.corr(), cmap='coolwarm', annot=True)
plt.title("Correlation Heatmap (Numeric Features Only)")
plt.show()

# 5. Model (drop AgeGroup for training)
X = df.drop(['Attrition', 'AgeGroup'], axis=1)
y = df['Attrition']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Logistic Regression with more iterations
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. SHAP Explainability
explainer = shap.LinearExplainer(model, X_train)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

# 7. Export for Power BI
# Ensure 'data' folder exists before saving
os.makedirs("data", exist_ok=True)
df['Predicted_Attrition'] = model.predict(X_scaled)
df.to_csv("data/attrition_results.csv", index=False)

print(" attrition_results.csv saved for Power BI")
