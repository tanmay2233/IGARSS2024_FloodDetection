import tifffile as tiff
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, generic_filter

# Load the TIF image and extract Channel 0
image_path = "./train/images/139.tif"

def local_variance(image, window_size):
    """Computes local variance in a window around each pixel."""
    mean_window = uniform_filter(image, window_size)
    squared_mean = uniform_filter(image ** 2, window_size)
    return squared_mean - mean_window ** 2  # Variance formula: E[x^2] - (E[x])^2

def lee_filter(image, window_size=5):
    """Applies Lee filter for speckle noise reduction."""
    mean_window = uniform_filter(image, window_size)
    variance_window = local_variance(image, window_size)
    overall_variance = np.var(image)
    
    # Compute Lee filter formula
    lee_filtered = mean_window + (variance_window / (variance_window + overall_variance)) * (image - mean_window)
    return lee_filtered

def frost_filter(image, window_size=5, damping_factor=1.5):
    """Applies Frost filter for speckle noise reduction while preserving edges."""
    rows, cols = image.shape
    frost_filtered = np.zeros_like(image, dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            # Define the local window
            r1, r2 = max(0, i - window_size // 2), min(rows, i + window_size // 2 + 1)
            c1, c2 = max(0, j - window_size // 2), min(cols, j + window_size // 2 + 1)
            local_window = image[r1:r2, c1:c2]

            # Compute local statistics
            local_mean = np.mean(local_window)
            local_variance = np.var(local_window)

            # Compute the Frost weight function
            k = np.exp(-damping_factor * np.abs(local_mean - image[i, j]) / (local_variance + 1e-6))
            frost_filtered[i, j] = k * local_mean + (1 - k) * image[i, j]

    return frost_filtered

def gamma_map_filter(image, window_size=5):
    """Applies Gamma MAP filter for speckle noise reduction."""
    mean_window = uniform_filter(image, window_size)
    variance_window = local_variance(image, window_size)
    overall_variance = np.var(image)

    # Compute gamma MAP filter
    gamma_filtered = mean_window + (variance_window / (variance_window + overall_variance)) * (image - mean_window)
    return gamma_filtered

try:
    # Read TIF image and extract Channel 0
    img = tiff.imread(image_path).astype(np.float32)
    channel_0 = img[:, :, 0]  

    # Normalize Channel 0 to range [0, 255]
    min_val, max_val = np.min(channel_0), np.max(channel_0)
    channel_0 = 255 * (channel_0 - min_val) / (max_val - min_val)
    channel_0 = channel_0.astype(np.uint8)  # Convert to uint8

    # Apply the filters
    lee_filtered = lee_filter(channel_0)
    frost_filtered = frost_filter(channel_0)
    gamma_filtered = gamma_map_filter(channel_0)

    # Convert results to uint8 for visualization
    lee_filtered = np.clip(lee_filtered, 0, 255).astype(np.uint8)
    frost_filtered = np.clip(frost_filtered, 0, 255).astype(np.uint8)
    gamma_filtered = np.clip(gamma_filtered, 0, 255).astype(np.uint8)

    # Display Original and Filtered Images
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = ["Original Image", "Lee Filter", "Frost Filter", "Gamma MAP Filter"]
    images = [channel_0, lee_filtered, frost_filtered, gamma_filtered]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.show()

except Exception as e:
    print(f"Error: {e}")
