import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# 1. Load the integrated data from Phase 3
# Ensure you are using the cleaned, integrated dataset
df = pd.read_csv('integrated_311_yelp (geospatial + semantic map).csv')

# 2. Select numerical features derived from Phase 3
features = ['proximity_score', 'category_similarity', 'hybrid_score', 'match_rank']
X = df[features].dropna()

# 3. Scale the features
# Scaling is essential for K-Means to ensure all scores contribute equally
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)




# 4. Final K-Means Fit
# Based on the elbow, select the best k (e.g., 4 or 5)
optimal_k = 4 
model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster_label'] = model.fit_predict(X_scaled)


# ---------------------------------------------------------
# ANALYSIS & FINDINGS (Phase 4 Requirements)
# ---------------------------------------------------------

# A. Visualizing Clusters using PCA (2D Projection)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['pca_1'] = X_pca[:, 0]
df['pca_2'] = X_pca[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='pca_1', y='pca_2', hue='cluster_label', palette='viridis', alpha=0.6)
plt.title('Complaint Type Clusters (PCA Reduced Dimensions)')
plt.show()

# B. Relationship between scores and clusters
# Profile each cluster by calculating the mean of its original scores
cluster_profile = df.groupby('cluster_label')[features].mean()
print("Cluster Analysis (Average Scores per Cluster):")
print(cluster_profile)
