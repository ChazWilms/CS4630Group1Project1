import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the cleaned, integrated data from Phase 3
df = pd.read_csv('integrated_311_yelp (geospatial + semantic map).csv')

# Select numerical features derived from Phase 3
features = ['proximity_score', 'category_similarity']
X = df[features].dropna()

# Standardize the data so that the same score in each variable is weighted the same
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



# K-means clustering

# Elbow Method to determine cluster count
inertia = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range, inertia, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method: Proximity vs Category Similarity')
plt.show()

# Final Model (Opted for k=4 for clearer categorization and interpretation)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
X['cluster'] = kmeans.fit_predict(X_scaled)



# Visualization and analysis

# Generate the Cluster Profile Chart for interpretation
cluster_profile = X.groupby('cluster').mean()
print("Cluster Profile Chart (Averages):")
print(cluster_profile)

# K-means cluster plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=X, x='proximity_score', y='category_similarity', 
                hue='cluster', palette='viridis', alpha=0.5)
plt.title('Clusters of Complaints by Proximity and Semantic Similarity')
plt.xlabel('Proximity Score (Geospatial)')
plt.ylabel('Category Similarity (Semantic)')
plt.legend(title='Cluster ID')
plt.show()
