import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset from CSV file
df = pd.read_csv('diabetes.csv')

# Use the entire dataset for clustering if you want to include all features
X = df.values

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# List of K values to try
k_values = [2, 3, 4]

# Loop through each K value
for k in k_values:
    # Initialize and fit K-Means model
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_scaled)
    y_kmeans = kmeans.predict(X_scaled)
    
    # Plotting the clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
    plt.title(f'K-Means Clustering (K={k})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.show()
