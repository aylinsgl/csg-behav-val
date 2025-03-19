"""
Utility functions
==========================================
Author: Aylin Kallmayer
Last updated: 2025-03-19
==========================================
Description:
This module provides utility functions for computing centroids from polygon coordinates
or binary masks, as well as a function to load image masks.
"""

import numpy as np
import cv2
from scipy.ndimage import center_of_mass

def calculate_centroid_x_y(polygon_x, polygon_y):
    """
    Calculate the centroid (center of mass) from polygon coordinates.

    Parameters
    ----------
    polygon_x : list or array-like
        The x-coordinates of the polygon vertices.
    polygon_y : list or array-like
        The y-coordinates of the polygon vertices.

    Returns
    -------
    tuple
        The (y, x) coordinates of the centroid.
    """
    # Combine x and y coordinates into a list of (x, y) tuples.
    polygon = np.array(list(zip(polygon_x, polygon_y)))
    # Create a binary mask with dimensions based on the maximum coordinates.
    mask = np.zeros((max(polygon_y) + 1, max(polygon_x) + 1), dtype=np.uint8)
    # Fill the polygon area in the mask.
    cv2.fillPoly(mask, [polygon], 1)
    # Compute the centroid using the center of mass.
    centroid = center_of_mass(mask)
    return centroid

def calculate_centroid_mask(mask):
    """
    Calculate the centroid (center of mass) from a given mask.

    The function thresholds the input mask to ensure it is binary, then computes the center of mass.

    Parameters
    ----------
    mask : numpy.ndarray
        The input image mask.

    Returns
    -------
    tuple
        The (y, x) coordinates of the centroid.
    """
    # Threshold the mask to obtain a binary mask.
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    # Compute the centroid using the center of mass.
    centroid = center_of_mass(binary_mask)
    return centroid

def load_mask(filename):
    """
    Load an image mask in grayscale.

    Parameters
    ----------
    filename : str
        Path to the image file.

    Returns
    -------
    numpy.ndarray
        The loaded mask as a grayscale image.
    """
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
