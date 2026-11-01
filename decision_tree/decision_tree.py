
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

filename = "Adult Census Income Dataset.csv"

df = pd.read_csv(filename)

print("Dataset loaded successfully!")
print("Dataset Shape:", df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values Before Replacing '?':")
print(df.isnull().sum())

df = df.replace("?", np.nan)

print("\nMissing Values After Replacing '?':")
print(df.isnull().sum())

df = df.dropna()

print("\nDataset Shape After Removing Missing Values:")
print(df.shape)

print("\nMissing Values After Removing Records:")
print(df.isnull().sum())

print("\nIncome Classes:")
print(df["income"].value_counts())

X = df.drop("income", axis=1)
y = df["income"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

categorical_columns = X.select_dtypes(
    include=["object"]
).columns

print("\nCategorical Columns:")
print(list(categorical_columns))

label_encoders = {}

for column in categorical_columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

print("\nCategorical features encoded successfully.")

income_encoder = LabelEncoder()

y = income_encoder.fit_transform(y)

print("\nIncome Classes:")
print(income_encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

dt_model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

print("\nDecision Tree model created successfully!")

dt_model.fit(X_train, y_train)

print("Decision Tree model trained successfully!")

y_pred = dt_model.predict(X_test)

print("\nPredictions:")
print(y_pred[:20])

comparison = pd.DataFrame({
    "Actual": y_test[:20],
    "Predicted": y_pred[:20]
})

print("\nActual vs Predicted:")
print(comparison)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("Accuracy (%):", accuracy * 100)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\nPrecision:", precision)
print("Precision (%):", precision * 100)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\nRecall:", recall)
print("Recall (%):", recall * 100)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\nF1-Score:", f1)
print("F1-Score (%):", f1 * 100)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=income_encoder.classes_,
        zero_division=0
    )
)

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(
    [0, 1],
    income_encoder.classes_
)

plt.yticks(
    [0, 1],
    income_encoder.classes_
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()
plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 10))

plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=income_encoder.classes_,
    filled=True,
    max_depth=3,
    fontsize=10
)

plt.title("Decision Tree - First 3 Levels")
plt.tight_layout()
plt.show()

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": dt_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)

plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance - Decision Tree")

plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

performance = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        f1
    ]
})

print("\nModel Performance:")
print(performance)

plt.figure(figsize=(8, 5))

plt.bar(
    performance["Metric"],
    performance["Score"]
)

plt.ylim(0, 1)

plt.xlabel("Evaluation Metric")
plt.ylabel("Score")

plt.title("Decision Tree Performance")

plt.tight_layout()
plt.show()
