import cv2
import numpy as np

# Load an example road image (replace with your own image)
image_path = 'right.jpg'  # Provide the correct path to your image
road_image = cv2.imread(image_path)

if road_image is None:
    print("Error: Could not load the image. Please check the file path.")
else:
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(road_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply Canny edge detection with adjusted parameters
    edges = cv2.Canny(blurred_image, 5, 500)  # The set values gives the best value for the images

    # Save the processed image
    success = cv2.imwrite('processed_image right.jpg', edges)

    if success:
        print("Processed image saved successfully as processed_image.jpg")
    else:
        print("Error: Could not save the processed image.")
