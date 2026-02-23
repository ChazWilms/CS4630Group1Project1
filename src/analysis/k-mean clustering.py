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

# ---------------------------------------------------------
# TECHNIQUE: K-MEANS CLUSTERING (Optimized for Scalability)
# ---------------------------------------------------------

# Find optimal clusters using the Elbow Method
inertia = []
K_range = range(2, 11)
for k in K_range:
    # n_init is set to 10 for consistency; random_state for reproducibility
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

# Visualization of the Elbow Plot
plt.figure(figsize=(8, 4))
plt.plot(K_range, inertia, 'bo-', markersize=8)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Sum of Squared Distances)')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()



# 4. Final K-Means Fit
# Based on the elbow, select the best k (e.g., 4 or 5)
optimal_k = 4 
model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster_label'] = model.fit_predict(X_scaled)
