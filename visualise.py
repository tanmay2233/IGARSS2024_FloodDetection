import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Paths to the image and ground truth
image_path = "./train/images/139.tif"
ground_truth_path = "./train/labels/139.png"

esa_worldcover_colors = {
    10: (0, 100, 0),        # Tree cover
    20: (255, 187, 34),     # Shrubland
    30: (255, 255, 76),     # Grassland
    40: (240, 150, 255),    # Cropland
    50: (250, 0, 0),        # Built-up
    60: (150, 150, 150),    # Bare / Sparse vegetation
    70: (255, 255, 255),    # Snow and Ice
    80: (0, 0, 255),        # Permanent Water Bodies
    90: (0, 207, 117),      # Herbaceous Wetland
    95: (0, 168, 89),       # Mangroves
    100: (255, 255, 255),   # Moss & Lichen
}

try:
    # Open the .tif image
    img = tiff.imread(image_path).astype(np.float32)

    # Check the shape and bit depth
    bit_depth = img.dtype.itemsize * 8  
    print(f"Image shape: {img.shape}")  # (Height, Width, Channels)
    print(f"Bit format: {bit_depth} bits per pixel")

    num_channels = img.shape[2]

    # Normalize channels 0 and 1 to [0, 255]
    for ch in [0, 1]:
        min_val, max_val = np.min(img[:, :, ch]), np.max(img[:, :, ch])
        img[:, :, ch] = 255 * (img[:, :, ch] - min_val) / (max_val - min_val)

    # Display all 6 channels in a 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle("All 6 Channels of the TIF Image", fontsize=14)

    for i in range(min(num_channels, 6)):  
        row, col = divmod(i, 3)

        if i == 4:  # Channel 5 (ESA WorldCover Map)
            world_cover = img[:, :, i].astype(np.uint8)

            # Create an RGB image with the ESA color map
            world_cover_rgb = np.zeros((world_cover.shape[0], world_cover.shape[1], 3), dtype=np.uint8)
            for key, color in esa_worldcover_colors.items():
                mask = world_cover == key
                world_cover_rgb[mask] = color

            axes[row, col].imshow(world_cover_rgb)
            axes[row, col].set_title("Channel 5 (ESA WorldCover)")
        else:
            axes[row, col].imshow(img[:, :, i], cmap="gray")
            axes[row, col].set_title(f"Channel {i + 1}")

        axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()

    if num_channels >= 6:
        channel_6 = img[:, :, 5].astype(np.uint8)

        # Count unique values and their frequency
        unique_values, counts = np.unique(channel_6, return_counts=True)

        # Print results
        print(f"\nChannel 6 has {len(unique_values)} unique values.")
        print("Unique values and their frequencies:")
        for val, freq in zip(unique_values, counts):
            print(f"Value: {val}, Frequency: {freq}")

    else:
        print("\nChannel 6 does not exist in this image.")

    # Open the ground truth PNG image
    ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)

    if ground_truth is None:
        print("\nError: Ground truth image not found or could not be loaded.")
    else:
        print(f"\nGround truth shape: {ground_truth.shape}")

        # Display the ground truth image
        plt.figure(figsize=(6, 6))
        plt.imshow(ground_truth, cmap="gray")
        plt.title("Ground Truth (PNG)")
        plt.axis("off")
        plt.show()

except Exception as e:
    print(f"Error: {e}")
