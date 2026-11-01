
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

file_path = "Wholesale customers data.csv"

df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

features = [
    'Fresh',
    'Milk',
    'Grocery',
    'Frozen',
    'Detergents_Paper',
    'Delicassen'
]

X = df[features]

print("\nFeatures used for clustering:")
print(features)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nData standardized successfully.")

silhouette_scores = []
inertia_values = []

K_values = range(2, 9)

for k in K_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    silhouette = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_scores.append(silhouette)
    inertia_values.append(kmeans.inertia_)

print("\nSilhouette Scores:")

for k, score in zip(K_values, silhouette_scores):
    print(
        "K =",
        k,
        " Silhouette Score =",
        round(score, 4)
    )

plt.figure(figsize=(8, 5))

plt.plot(
    K_values,
    inertia_values,
    marker='o'
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.grid()

plt.show()

plt.figure(figsize=(8, 5))

plt.plot(
    K_values,
    silhouette_scores,
    marker='o'
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score Analysis")

plt.grid()

plt.show()

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

df['Cluster'] = clusters

print("\nNumber of customers in each cluster:")

print(
    df['Cluster']
    .value_counts()
    .sort_index()
)

cluster_summary = df.groupby('Cluster')[features].mean()

print("\nAverage Spending of Customers in Each Cluster:")

print(
    cluster_summary.round(2)
)

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(9, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=df['Cluster'],
    s=50
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Customer Segmentation using K-Means")

plt.grid()

plt.show()

cluster_summary.plot(
    kind='bar',
    figsize=(12, 6)
)

plt.xlabel("Cluster")
plt.ylabel("Average Annual Spending")
plt.title("Average Spending by Customer Cluster")

plt.xticks(rotation=0)

plt.legend()

plt.grid(axis='y')

plt.show()

print("\nFinal Dataset with Cluster Labels:")

print(df.head(10))

print("\nCluster Size:")

for cluster in sorted(df['Cluster'].unique()):

    count = (
        df['Cluster'] == cluster
    ).sum()

    print(
        "Cluster",
        cluster,
        ":",
        count,
        "customers"
    )

final_score = silhouette_score(
    X_scaled,
    df['Cluster']
)

print(
    "\nFinal Silhouette Score for K = 3:",
    round(final_score, 4)
)

print("\nConclusion:")

print(
    "K-Means clustering was successfully applied "
    "to the Wholesale Customers Dataset."
)

print(
    "Three customer clusters were obtained based "
    "on their annual spending patterns."
)

print(
    "The clusters can be used for customer "
    "segmentation, targeted marketing and "
    "delivery planning."
)
