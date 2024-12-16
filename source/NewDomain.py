# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 22:47:32 2024

@author: lenovo
"""

import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tkinter import Tk, filedialog
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_npy_files(folder):
    """Get all .npy file paths in the folder."""
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npy')]

def load_npy_data(file_paths):
    """Load .npy files and merge them into one array."""
    data_list = [np.load(file) for file in file_paths]
    return np.vstack(data_list)

def compute_centroid(vectors):
    """Calculate the centroid of 3D vectors."""
    return np.mean(vectors, axis=0)

def perform_pca(data, n_components=3):
    """Use PCA to reduce dimensionality to three dimensions."""
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)

def main(output_callback):
    # Create window
    Tk().withdraw()  # Hide the main window

    embeddings_list = []  # Store embeddings from all folders
    folder_paths = []  # Store folder paths for coloring
    all_centroids = []  # Store centroids for each folder

    # Select multiple folders and load data
    while True:
        folder_path = filedialog.askdirectory(title="Select Folder")  # Choose folder
        if not folder_path:
            break

        npy_files = get_npy_files(folder_path)
        if not npy_files:
            continue

        # Load and store data
        data = load_npy_data(npy_files)
        embeddings_list.append(data)
        folder_paths.append(folder_path)

    if not embeddings_list:
        print("No valid data loaded.")
        return

    # Combine all embeddings and standardize data
    all_embeddings = np.vstack(embeddings_list)
    scaler = StandardScaler().fit(all_embeddings)
    all_embeddings = scaler.transform(all_embeddings)

    # Perform PCA on the combined data
    pca_results = perform_pca(all_embeddings)

    # Compute centroids for each folder
    start_idx = 0
    for data in embeddings_list:
        num_embeddings = len(data)
        pca_subset = pca_results[start_idx:start_idx + num_embeddings]
        centroid = compute_centroid(pca_subset)
        all_centroids.append(centroid)
        start_idx += num_embeddings

    # Plot the 3D PCA results
    fig = plt.figure(figsize=(10, 8))
    fig.subplots_adjust(right=0.7)  # Adjust layout for legend
    ax = fig.add_subplot(111, projection='3d')
    color_map = plt.cm.get_cmap('tab10', len(folder_paths))

    start_idx = 0
    for i, folder_path in enumerate(folder_paths):
        embeddings = embeddings_list[i]
        num_embeddings = len(embeddings)
        pca_subset = pca_results[start_idx:start_idx + num_embeddings]
        ax.scatter(pca_subset[:, 0], pca_subset[:, 1], pca_subset[:, 2], color=color_map(i), label=os.path.basename(folder_path))
        start_idx += num_embeddings

    ax.set_title('3D PCA of Embeddings')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))  # Move legend outside the plot

    # Select user's vector
    new_vector_file = filedialog.askopenfilename(title="Select New .npy File", filetypes=[("Numpy files", "*.npy")])
    
    if new_vector_file:
        new_vector = np.load(new_vector_file)

        # Standardize new vector based on the original data
        new_vector_standardized = scaler.transform(new_vector.reshape(1, -1))

        # Combine with original data and apply PCA
        combined_data = np.vstack([all_embeddings, new_vector_standardized])
        three_d_combined_vectors = perform_pca(combined_data)

        # Extract the 3D representation of the new vector
        new_vector_3d = three_d_combined_vectors[-1]

        # Plot the new vector in a different color
        ax.scatter(new_vector_3d[0], new_vector_3d[1], new_vector_3d[2], color='red', s=100, label='New Vector')

        # Calculate and output Euclidean distance to all centroids
        for idx, centroid in enumerate(all_centroids):
            distance = euclidean(new_vector_3d, centroid)
            print(f"Distance to folder {idx + 1}: {distance}")
            output_callback(f"Distance to folder {idx + 1}: {distance}")

    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))  # Update legend to include the new vector
    plt.show()  # Show the plot
    return fig
