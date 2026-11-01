import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

filename = "Adult Census Income Dataset.csv"

df = pd.read_csv(filename)

print("First 5 records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

df.columns = df.columns.str.strip()

for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].str.strip()

target_column = "income"

if target_column not in df.columns:
    print("Available columns:")
    print(df.columns.tolist())
    raise ValueError("Target column 'income' was not found.")

print("\nTarget Column:", target_column)

df = df.replace("?", np.nan)

print("\nMissing Values After Replacing '?':")
print(df.isnull().sum())

df = df.dropna()

print("\nDataset Shape After Removing Missing Values:")
print(df.shape)

X = df.drop(target_column, axis=1)
y = df[target_column]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget Classes:")
print(y.unique())

X = pd.get_dummies(X, drop_first=True)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("\nEncoded Feature Shape:")
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

print("\nRandom Forest Model Training Completed.")

y_pred = rf_model.predict(X_test)

print("\nFirst 20 Predictions:")
print(y_pred[:20])

accuracy = accuracy_score(y_test, y_pred)

print("\nRandom Forest Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

disp.plot()
plt.title("Random Forest - Confusion Matrix")
plt.show()

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 Important Features:")
print(feature_importance.head(15).to_string(index=False))

top_features = feature_importance.head(15)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 15 Feature Importances - Random Forest")
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

print("\n================ PERFORMANCE SUMMARY ================")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("======================================================")