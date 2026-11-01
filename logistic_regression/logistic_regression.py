
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("Titanic-Dataset.csv")

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])
df["Embarked"] = le.fit_transform(df["Embarked"])

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy =", accuracy_score(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

X_age = df[["Age"]]
y_age = df["Survived"]

graph_model = LogisticRegression()
graph_model.fit(X_age, y_age)

age_range = pd.DataFrame({
    "Age": np.linspace(df["Age"].min(), df["Age"].max(), 300)
})

probability = graph_model.predict_proba(age_range)[:, 1]

plt.figure(figsize=(8,6))

plt.scatter(
    df["Age"],
    y_age,
    color="blue",
    alpha=0.4,
    label="Actual Data"
)

plt.plot(
    age_range["Age"],
    probability,
    color="red",
    linewidth=3,
    label="Logistic Regression Curve"
)

plt.xlabel("Age")
plt.ylabel("Probability of Survival")
plt.title("Logistic Regression on Titanic Dataset")
plt.legend()
plt.grid(True)

plt.show()
