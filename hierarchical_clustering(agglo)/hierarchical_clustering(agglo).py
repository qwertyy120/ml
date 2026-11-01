import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

df = pd.read_csv("Adult Census Income Dataset.csv")

print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Records:")
print(df.head())

features = [
    'age',
    'fnlwgt',
    'education.num',
    'capital.gain',
    'capital.loss',
    'hours.per.week'
]

X = df[features].copy()

X = X.replace('?', np.nan)
X = X.dropna()

print("\nData after removing missing values:")
print(X.shape)

X_sample = X.sample(n=20, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)

Z = linkage(
    X_scaled,
    method='single',
    metric='euclidean'
)

plt.figure(figsize=(12, 6))

dendrogram(
    Z,
    truncate_mode='lastp',
    p=30
)

plt.title("Hierarchical Clustering using Single Linkage")
plt.xlabel("Clusters / Data Points")
plt.ylabel("Euclidean Distance")

plt.show()

model = AgglomerativeClustering(
    n_clusters=3,
    metric='euclidean',
    linkage='single'
)

labels = model.fit_predict(X_scaled)

result = X_sample.copy()
result['Cluster'] = labels

print("\nCluster Distribution:")
print(
    result['Cluster']
    .value_counts()
    .sort_index()
)

print("\nClustered Data:")
print(result.head(10))

print("\nCluster-wise Mean:")
print(
    result.groupby('Cluster')[features].mean()
)

print("\nSilhouette Score:", round(silhouette_score(X_scaled, labels), 4))
