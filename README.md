Defect Detection and Analysis System
  This repository provides a Python-based defect detection and analysis system that includes interactive segmentation, feature extraction, and domain adaptation. The code leverages PyQt5 for GUI implementation, OpenCV for image processing, scikit-learn for PCA-based dimensionality reduction, and Matplotlib for embedding visualizations. Each file in the repository contributes to a step-by-step workflow for detecting, analyzing, and comparing defect features across multiple domains.



Table of Contents
  Overview
  Installation
  File Descriptions
    Main GUI (ui_dialog.py)
    Interactive Segmentation (segmentation.py)
    Feature Extraction (feature_extraction.py)
    Embedding Visualization (embedding_visu.py)
    New Domain Embedding Comparison (new_domain.py)
    Text Content (TextContent.json)
  Usage
  Acknowledgments



Overview
  This system is designed to assist in the interactive segmentation, feature extraction, and analysis of defects in images. The GUI allows users to load images, segment regions of interest, extract features, and visualize embeddings with dimensionality reduction techniques. Additionally, it enables the comparison of new defect features with existing domains, offering insight into feature similarities across different defect datasets.



Installation
1. Clone this repository:

  git clone https://github.com/your-username/defect-detection-system.git

2. Navigate to the project directory:

  cd defect-detection-system

3. Install required packages:
  Ensure Python 3.8 or newer is installed.
  Install dependencies:

    pip install -r requirements.txt



File Descriptions:
  Main GUI (ui_dialog.py)
    This script implements the primary GUI for the system using PyQt5. Key features include:

      Image Display: Supports displaying the original and segmented images.
      Segmentation and Feature Extraction: Runs segmentation and feature extraction scripts directly from the GUI.
      Embedding Visualization: Launches embedding visualizations, with options to compare new domains.
      Dynamic Text Loading: Loads instructional and descriptive text from TextContent.json to guide the user.
  
  Interactive Segmentation (segmentation.py)
    The segmentation.py file contains functions that allow users to interactively segment a region of interest (ROI) in an image. This script enables users to:

      1. Select a bounding box around the defect area.
      2. Refine the mask by marking foreground and background areas.
      3. Apply the GrabCut algorithm for precise segmentation.
      4.Save the segmented image in the same directory as the input image for further processing.

  Feature Extraction (feature_extraction.py)
    This script is dedicated to feature extraction for the segmented defects, including:

      1. Shape Features: Derived from contour and Fourier descriptors.
      2. Color Histogram Features: Extracted from the HSV color space.
      3. Texture Features: Calculated using GLCM, entropy, LBP, Gabor filter, and wavelet transforms.
      4. Edge Features: Extracted using Sobel operator for gradient-based measurements.
      5. Output: The extracted features are saved as .txt and .npy files, making them accessible for embedding visualization and analysis.

  Embedding Visualization (embedding_visu.py)
    The embedding_visu.py file implements a 3D PCA-based visualization of embeddings. It includes:

      Loading Embeddings: Embeddings are loaded from multiple directories, allowing domain-based visual comparisons.
      Dimensionality Reduction: Uses PCA to reduce embeddings to 3D space for better visualization.
      Interactive Visualization: Displays embeddings in a 3D plot, color-coded by domain.
      Returning a Plot Object: For GUI integration, the plot can be embedded in PyQt5 using Matplotlib.

  New Domain Embedding Comparison (new_domain.py)
    This script allows for the addition and comparison of a new domain embedding. Features include:

      1. PCA Projection: The new vector and existing embeddings are projected into 3D space using PCA.
      2. Euclidean Distance Calculation: Computes distances from the new vector to each existing domain centroid to quantify similarity.
      3. Display in GUI: Allows the user to view distance values and assess the relative closeness of new features to existing domains.

  Text Content (TextContent.json)
    This JSON file stores text content for display in the GUI. It includes user guidance for various sections, ensuring a smooth experience by providing instructions for each feature.



Usage
  1. Launch the Application:
    Run ui_dialog.py as the main entry point to start the PyQt5 GUI.

      python ui_dialog.py

  2. Load and Segment an Image:
    Use the GUI to load an image and interactively segment the ROI using bounding boxes and brush tools.

  3. Extract and Save Features:
    Click to extract and save shape, color, texture, and edge features. The extracted data is saved for embedding visualization.

  4. Visualize Embeddings:
    Choose directories containing .npy files of embeddings. The embeddings are then visualized in a 3D PCA plot.

  5. New Domain Comparison:
    Add a new embedding for comparison with existing domains. View the calculated distances to assess similarity.



Acknowledgments
  This project utilizes PyQt5, OpenCV, scikit-learn, and Matplotlib. Special thanks to contributors who developed foundational libraries and documentation that supported the development of this project.
