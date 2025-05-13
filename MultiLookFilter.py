import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import view_as_blocks

# Load the TIF image and extract Channel 0
image_path = "./train/images/139.tif"

def multi_look(image, n_looks=4):
    """
    Apply Multi-Looking by averaging non-overlapping sub-blocks of the image.
    
    Args:
        image (numpy.ndarray): Input SAR image (single channel).
        n_looks (int): Number of looks (determines block size).

    Returns:
        numpy.ndarray: Multi-looked image with reduced speckle.
    """
    height, width = image.shape
    block_size = int(np.sqrt(n_looks))  # Define block size based on number of looks

    # Ensure block size fits into image dimensions
    new_height = (height // block_size) * block_size
    new_width = (width // block_size) * block_size
    cropped_image = image[:new_height, :new_width]

    # Divide the image into non-overlapping blocks
    blocks = view_as_blocks(cropped_image, block_shape=(block_size, block_size))

    # Compute mean of each block to reduce speckle
    averaged_blocks = blocks.mean(axis=(2, 3))

    # Resize back to original dimensions
    multi_looked_image = np.kron(averaged_blocks, np.ones((block_size, block_size)))

    return multi_looked_image

try:
    # Read TIF image and extract Channel 0
    img = tiff.imread(image_path).astype(np.float32)
    channel_0 = img[:, :, 0]  

    # Normalize Channel 0 to range [0, 255]
    min_val, max_val = np.min(channel_0), np.max(channel_0)
    channel_0 = 255 * (channel_0 - min_val) / (max_val - min_val)
    channel_0 = channel_0.astype(np.uint8)

    # Apply Multi-Looking filter
    multi_looked = multi_look(channel_0, n_looks=4)

    # Convert to uint8 for visualization
    multi_looked = np.clip(multi_looked, 0, 255).astype(np.uint8)

    # Display Original and Multi-Looked Image
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    titles = ["Original Channel 0", "Multi-Looked Image (Speckle Reduced)"]
    images = [channel_0, multi_looked]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.show()

except Exception as e:
    print(f"Error: {e}")
