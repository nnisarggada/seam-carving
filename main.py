import cv2
import numpy as np


# Compute the energy map using the gradient of the image
def calculate_energy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    energy = np.abs(sobel_x) + np.abs(sobel_y)
    return energy


# Find the vertical seam to remove using dynamic programming
def find_seam(energy_map):
    rows, cols = energy_map.shape
    seam = np.zeros(rows, dtype=np.uint32)
    cost = np.zeros((rows, cols))
    cost[0, :] = energy_map[0, :]

    # Compute the cumulative energy map
    for i in range(1, rows):
        for j in range(0, cols):
            min_cost = cost[i - 1, j]
            if j > 0:
                min_cost = min(min_cost, cost[i - 1, j - 1])
            if j < cols - 1:
                min_cost = min(min_cost, cost[i - 1, j + 1])
            cost[i, j] = energy_map[i, j] + min_cost

    # Find the end of the minimum seam path
    seam[-1] = np.argmin(cost[-1])

    # Trace back the seam
    for i in range(rows - 2, -1, -1):
        prev_x = seam[i + 1]

        # Define a range around the previous seam position, ensuring within bounds
        start = max(0, prev_x - 1)
        end = min(cols, prev_x + 2)

        # Calculate the offset within the safe range
        offset = np.argmin(cost[i, start:end]) + start
        seam[i] = offset

    return seam


# Remove the identified seam from the image
def remove_seam(image, seam):
    rows, cols, _ = image.shape
    output = np.zeros((rows, cols - 1, 3), dtype=np.uint8)
    for i in range(rows):
        output[i, :, 0] = np.delete(image[i, :, 0], seam[i])
        output[i, :, 1] = np.delete(image[i, :, 1], seam[i])
        output[i, :, 2] = np.delete(image[i, :, 2], seam[i])
    return output


# Seam carving function to reduce image size by removing seams
def seam_carve(image, num_seams):
    for _ in range(num_seams):
        energy_map = calculate_energy(image)
        seam = find_seam(energy_map)
        image = remove_seam(image, seam)
    return image


# Load the image
image = cv2.imread("input.jpg")

# Get the current width of the image
current_width = image.shape[1]
print(f"Current width of the image: {current_width}")

# Ask user for the new width
new_width = int(input("Enter the desired width: "))

# Calculate the number of seams to remove
num_seams = current_width - new_width
if num_seams < 0:
    print("The new width should be less than the current width.")
else:
    # Apply seam carving
    output_image = seam_carve(image, num_seams)

    # Save and show the output image
    cv2.imwrite("output.jpg", output_image)
    cv2.imshow("Seam Carved Image", output_image)
    print("Seam carved image saved as output.jpg")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
