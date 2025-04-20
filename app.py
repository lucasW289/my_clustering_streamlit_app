# app.py
import streamlit as st
import pickle
import matplotlib.pyplot as plt

# Load the model AND the training data
with open('kmeans_model.pkl', 'rb') as f:
    loaded_model, X = pickle.load(f)

# Streamlit page setup
st.set_page_config(page_title="k-Means Clustering App", layout="centered")

# App title and info
st.title("k-Means Clustering Visualizer")
st.markdown("**By Aung Phyo Linn - 6531501204**")

# Description
st.subheader("📊 Visualization Using Training Data")
st.markdown("This demo shows the same data that was used to train the k-Means model, so the clusters match exactly.")

# Predict using loaded model
y_kmeans = loaded_model.predict(X)

# Plotting
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', s=50, label='Data Points')
ax.scatter(loaded_model.cluster_centers_[:, 0], loaded_model.cluster_centers_[:, 1], 
           c='red', s=200, alpha=0.75, marker='X', label='Centroids')
ax
