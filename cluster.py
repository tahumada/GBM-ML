import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

latent_feats = np.load('latent_feats.npy')
burst_list = np.load('burst_list.npy')

# Fit Bayesian Gaussian Mixture Model
bgm = GaussianMixture(n_components=2).fit(latent_feats)
cluster_assignments = bgm.predict(latent_feats)

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, perplexity=50, metric='l1')
latent_feats_2d = tsne.fit_transform(latent_feats)

# Fit Isolation Forest to find outliers
iso_forest = IsolationForest()
outlier_labels = iso_forest.fit_predict(latent_feats)

# Combining Cluster and Outlier Labels:
# Create a DataFrame with the cluster assignments.
df = pd.DataFrame({
    'burst_id': burst_list.astype(int),
    'cluster_label': cluster_assignments
})

# Identify and set outliers to label -1. This is the crucial step.
df['cluster_label'] = np.where(outlier_labels == -1, -1, df['cluster_label'])

# --- Save to CSV ---
df.to_csv('cluster_labels.csv', index=False)

# This section is the historic t-SNE plotting.
# We found it to be more salient to use PCA, however I figured t-SNE could still be of use.
'''
# Visualize clusters and outliers
plt.figure(figsize=(10, 8))
for cluster in np.unique(cluster_assignments):
    cluster_points = latent_feats_2d[cluster_assignments == cluster]
    if not cluster_points.size == 0:
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}', s=5)
outliers = latent_feats_2d[outlier_labels == -1]
plt.scatter(outliers[:, 0], outliers[:, 1], c='red', s=5, label='Outliers')
plt.title('Clusters and Outliers in Latent Space')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.show()
plt.savefig('clusters_outliers.png')
'''

# Identify and print and save burst IDs of outliers
outlier_burst_ids = burst_list[outlier_labels == -1]
outliers_df = pd.DataFrame({'burst_id': outlier_burst_ids.astype(int)})
outliers_df.to_csv('outlier_burst_ids.csv', index=False)


# Construct and project the PCA (alternative to t-SNE)
pca = PCA(n_components=2)
latent_feats_pca = pca.fit_transform(latent_feats)

for cluster in np.unique(cluster_assignments):
    cluster_points = latent_feats_pca[cluster_assignments == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}', s=5)
    plt.scatter(latent_feats_pca[outlier_labels == -1][:, 0], latent_feats_pca[outlier_labels == -1][:, 1], 
                c='red', s=5, label='Outliers')
plt.title('PCA Projection')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()

plt.tight_layout()
plt.show()
plt.savefig('PCA.png')