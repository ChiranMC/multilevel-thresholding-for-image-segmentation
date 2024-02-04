import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_entropy(probabilities):
    return -np.sum([p * np.log2(p + 1e-10) for p in probabilities if p > 0])

def calculate_total_information_entropy(p_background, p_object, h_background, h_object):
    return -(p_background * h_background + p_object * h_object)

def custom_multilevel_thresholding_mec(image, num_levels):
    # Converting the original image to grayscale image 
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding using Gaussian mean
    binary_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Calculating initial thresholds
    initial_thresholds = np.linspace(0, 255, num_levels + 2)[1:-1]

    # Refine thresholds using Maximum entropy criterion (MEC)
    for _ in range(10):  # can adjust the number of iterations if want
        updated_thresholds = []
        for i in range(num_levels - 1):
            region = binary_image[
                (binary_image >= initial_thresholds[i]) & (binary_image <= initial_thresholds[i + 1])
            ]
            if region.size > 0:  # Check if the region is not empty
                probabilities = region.flatten() / 255.0
                p_object = np.sum(probabilities)
                p_background = 1 - p_object
                h_object = calculate_entropy(probabilities)
                h_background = calculate_entropy(1 - probabilities)

                total_information_entropy = calculate_total_information_entropy(p_background, p_object, h_background, h_object)

                updated_thresholds.append(total_information_entropy)

        # Update initial thresholds
        if updated_thresholds:
            best_threshold_index = np.argmax(updated_thresholds)
            best_threshold = (initial_thresholds[best_threshold_index] + initial_thresholds[best_threshold_index + 1]) / 2.0
            initial_thresholds[1:-1] = best_threshold

    # Apply the final thresholds
    result_image = np.zeros_like(gray_image)
    for i in range(num_levels - 1):
        result_image[
            (gray_image >= initial_thresholds[i]) & (gray_image <= initial_thresholds[i + 1])
        ] = 255 * (i + 1) // num_levels

    return result_image, initial_thresholds

# Allow the user to input the image path
image_path = input("Enter the path to the image: ")

# Load an image with error handling
loaded_image = cv2.imread(image_path)
if loaded_image is None:
    print(f"Error: Unable to load image from {image_path}")
    exit()

# Specify the number of levels for multilevel thresholding within the range of 1 to 255
num_levels_input = int(input("Enter the number of levels for multilevel thresholding (1 to 255): "))
num_levels_clipped = np.clip(num_levels_input, 1, 255)  # Ensure the value is within the valid range

# Perform multilevel thresholding using MEC and get thresholds
processed_image_mec, calculated_thresholds_mec = custom_multilevel_thresholding_mec(loaded_image, num_levels_clipped)

# Plot the original image, multilevel thresholded image, and histogram in a single figure for MEC
plt.figure(figsize=(10, 8))

# Original Image
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

# Multilevel Thresholded Image using MEC
plt.subplot(2, 2, 2)
plt.imshow(processed_image_mec, cmap='gray')
plt.title("Multilevel Thresholded Image (MEC)")
plt.axis("off")

# Histogram for Multilevel Thresholded Image using MEC
plt.subplot(2, 2, 3)
plt.title("Histogram for Multilevel Thresholded Image (MEC)")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.hist(processed_image_mec.flatten(), bins=num_levels_clipped, range=[0, 256], color='black', histtype='step')

plt.show()

