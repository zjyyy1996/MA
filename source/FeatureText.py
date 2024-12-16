# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 22:11:01 2024

@author: jingyan
"""

import argparse
import cv2 
import numpy as np
import os
from scipy import stats
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import img_as_ubyte
from scipy.fft import fft
import math
from skimage.filters import gabor
import pywt
from tkinter import Tk, filedialog


# Crop image function
def crop_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found in the image.")
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

# Image preprocessing: grayscale conversion and binarization
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
    return binary

# Calculate shape parameters using contour feature method and Fourier descriptor method
def calculate_shape_parameters(image):
    binary = preprocess_image(image)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found in the image.")
        return np.zeros(14)  # Return a zero array if no contours are found

    c = max(contours, key=cv2.contourArea)

    # Basic moment features
    moments = cv2.moments(c)
    hu_moments = cv2.HuMoments(moments).flatten()

    # 1. Aspect ratio
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = float(w) / h if h != 0 else 0

    # 2. Rectangular extent
    area = cv2.contourArea(c)
    rect_area = w * h
    extent = area / rect_area if rect_area != 0 else 0

    # 3. Circularity
    perimeter = cv2.arcLength(c, True)
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter != 0 else 0

    # 4. Convexity
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    convexity = area / hull_area if hull_area != 0 else 0

    # 5. Solidity (same as convexity here)
    solidity = area / hull_area if hull_area != 0 else 0

    # 6. Equivalent diameter
    (x, y), radius = cv2.minEnclosingCircle(c)
    equivalent_diameter = 2 * radius

    # 7. Major axis ratio and eccentricity
    major_axis_length = minor_axis_length = 0  # Initialize major and minor axis lengths
    if len(c) >= 5:
        ellipse = cv2.fitEllipse(c)
        major_axis_length = max(ellipse[1])
        minor_axis_length = min(ellipse[1])
        axis_ratio = major_axis_length / minor_axis_length if minor_axis_length != 0 else 0
        eccentricity = np.sqrt(1 - (minor_axis_length / major_axis_length) ** 2)
    else:
        axis_ratio = 0
        eccentricity = 0

    # 8. Roundness
    roundness = (4 * area) / (np.pi * (major_axis_length / 2) ** 2) if major_axis_length != 0 else 0

    # Combine all feature vectors
    features = np.concatenate(([aspect_ratio, extent, circularity, convexity, solidity, equivalent_diameter, axis_ratio, roundness, eccentricity], hu_moments))

    ###print("Shape features (final):", features)
    return features

# Define a simplified color histogram feature extraction function
def calculate_simple_color_histogram(image, num_bins=4):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_channel = image_hsv[:, :, 0]  # Extract H channel
    hist = cv2.calcHist([h_channel], [0], None, [num_bins], [0, 180])
    hist = cv2.normalize(hist, hist).flatten()
    ###print("Color features (HSV histogram):", hist)  # Display color feature information
    return np.array(hist)

# Calculate texture features using GLCM, entropy, LBP, Gabor filter, and Wavelet transform
def calculate_texture_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # GLCM texture features
    glcm = graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
    contrast = np.mean(graycoprops(glcm, 'contrast'))
    homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
    energy = np.mean(graycoprops(glcm, 'energy'))
    correlation = np.mean(graycoprops(glcm, 'correlation'))
    dissimilarity = np.mean(graycoprops(glcm, 'dissimilarity'))

    # Entropy feature
    entropy_value = entropy(img_as_ubyte(gray), disk(5))
    avg_entropy = np.mean(entropy_value)

    # LBP texture feature
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))

    # Gabor features
    gabor_features = []
    frequencies = [0.1, 0.2, 0.3]
    orientations = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    for freq in frequencies:
        for theta in orientations:
            real, _ = gabor(gray, frequency=freq, theta=theta)
            gabor_features.append(real.mean())
            gabor_features.append(real.std())

    # Wavelet features
    coeffs = pywt.wavedec2(gray, 'haar', level=2)
    wavelet_features = []
    for coeff in coeffs:
        if isinstance(coeff, tuple):
            for arr in coeff:
                wavelet_features.append(arr.mean())
                wavelet_features.append(arr.std())
        else:
            wavelet_features.append(coeff.mean())
            wavelet_features.append(coeff.std())

    # Combine all texture features
    texture_features = np.concatenate((
        [contrast, homogeneity, energy, correlation, dissimilarity, avg_entropy],
        lbp_hist,
        gabor_features,
        wavelet_features
    )).flatten()

    ###print("Texture features:", texture_features, "\nLength:", len(texture_features))
    return texture_features

# Calculate edge features using edge detection methods
def calculate_edge_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    mean_edge_intensity = np.mean(edge_magnitude)
    std_edge_intensity = np.std(edge_magnitude)
    min_edge_intensity = np.min(edge_magnitude)
    max_edge_intensity = np.max(edge_magnitude)
    edge_features = np.array([mean_edge_intensity, std_edge_intensity, min_edge_intensity, max_edge_intensity]).flatten()
    ###print("Edge features:", edge_features, "\nLength:", len(edge_features))
    return edge_features

def main():
    
    # Create the parser
    parser = argparse.ArgumentParser(description="Print the input argument to the console.")
    
    # Add an argument for the input image path
    parser.add_argument("input_argument", type=str, help="The input argument to be printed")
    
    # Parse the arguments
    args = parser.parse_args()

    # Get the file path from the input argument instead of user input
    img_path = args.input_argument

    # Read the image
    image = cv2.imread(img_path)
    if image is None:
        print(f"Cannot read image or image does not exist: {img_path}")
        return
    
    save_dir = os.path.dirname(img_path)
    img_name = os.path.basename(img_path)
    img_base_name, img_ext = os.path.splitext(img_name)
    save_txt_path = os.path.join(save_dir, f"{img_base_name}_feature.txt")
    save_npy_path = os.path.join(save_dir, f"{img_base_name}_feature.npy")
   
    cropped_image = crop_image(image)
    if cropped_image is None:
        print("Cannot crop the image.")
        return

    save_cropped_image_path = os.path.join(save_dir, f"{img_base_name}_cropped{img_ext}")
    success = cv2.imwrite(save_cropped_image_path, cropped_image)
    if not success:
        print(f"Failed to save cropped image: {save_cropped_image_path}")
        return

    try:
        shape_parameters = calculate_shape_parameters(cropped_image)
        color_features = calculate_simple_color_histogram(cropped_image)  # Display color feature information
        texture_features = calculate_texture_features(cropped_image)
        edge_features = calculate_edge_features(cropped_image)
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        return

    # Combine all features and save them
    all_features = np.concatenate((shape_parameters, color_features, texture_features, edge_features)).flatten()
    all_features = np.nan_to_num(all_features, nan=0.0)
    result_text = "\n".join([f"{feature:.4f}" for feature in all_features])
    print(all_features)

    # Save to .txt file
    with open(save_txt_path, 'w') as f:
        f.write(result_text)
    ###print(f"Features saved to {save_txt_path}")

    # Save to .npy file
    np.save(save_npy_path, all_features)
    ###print(f"Features saved to {save_npy_path}")

if __name__ == "__main__":
    main()
