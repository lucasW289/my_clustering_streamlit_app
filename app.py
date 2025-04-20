# Author: Nongnuch

import streamlit as st
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Load the trained k-means model
with open('kmeans_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Set the page config
st.set_page_config(page_title="k-Means Clustering App", layout="centered")

# Set title
st.title("k-Means Clustering Visualizer")

# Subheader
st.subheader("Example Data for Visualization")
st.markdown("This demo uses example data (2D) to illustrate clustering results.")

# Generate synthetic data (2D)
X, _ = make_blobs(n_samples=300, centers=loaded_model.n_clusters, cluster_std=0.60, random_state=0)

# Predict cluster labels using the loaded model
y_kmeans = loaded_model.predict(X)

# Plotting
fig, ax = plt.subplots()
scatter = ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

# Plot cluster centers
centers = loaded_model.cluster_centers_
ax.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.75, marker='X', label='Centroids')

ax.set_title("Cluster Visualization")
ax.legend()

# Display plot in Streamlit
st.pyplot(fig)
