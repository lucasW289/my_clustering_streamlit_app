import streamlit as st
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Load the trained model
with open('kmeans_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Streamlit page config
st.set_page_config(page_title="k-Means Clustering App", layout="centered")

# App title
st.title("k-Means Clustering Visualizer")
st.markdown("**By Aung Phyo Linn - 6531501204**")

# Description
st.subheader("📊 Example Data for Visualization")
st.markdown("This demo uses example 2D synthetic data to illustrate clustering results using a pre-trained k-Means model.")

# Generate synthetic 2D data for visualization
X, _ = make_blobs(n_samples=300, centers=loaded_model.n_clusters, cluster_std=0.60, random_state=42)

# Predict cluster labels using the loaded model
y_kmeans = loaded_model.predict(X)

# Create matplotlib figure
fig, ax = plt.subplots()
scatter = ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', s=50, label='Data Points')

# Plot cluster centers
centers = loaded_model.cluster_centers_
ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Centroids')

# Add legend and title
ax.legend()
ax.set_title("k-Means Clustering Result")

# Display the plot in Streamlit
st.pyplot(fig)
