import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
import joblib

# Function to extract features from a given image
def extract_features_from_image(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Use edge detection techniques
    edges = cv2.Canny(blurred_image, 50, 150)

    # Find contours in the image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small or irrelevant contours
    min_contour_area = 100
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Extract features for each contour
    features = [cv2.contourArea(cnt) for cnt in filtered_contours]

    return features

# Specify the path to your dataset folder
dataset_folder = 'pics'

# Load a single image from the dataset
# filename = os.listdir(dataset_folder)[0]
# image_path = os.path.join(dataset_folder, filename)
# img = cv2.imread(image_path)

# Check if the image is loaded successfully

# get the list of images 
filename = os.listdir(dataset_folder)
# iteratre thru the list for every image 
for img_name in filename:
    # get the image path 
    image_path = os.path.join(dataset_folder, img_name)
    # read the image 
    img = cv2.imread(image_path)

    if img is not None:
        # Print the loaded image for debugging
        print("Loaded image:", image_path)

        # Extract features from the image
        all_features = extract_features_from_image(img)

        # Convert the feature list to a numpy array
        features_array = np.array(all_features).reshape(-1, 1)

        # Use k-means clustering to identify parking spots
        n_clusters = 2  # You may adjust this based on the number of parking spot classes you expect
        n_init_value = 10  # Explicitly set the value for n_init
        kmeans = KMeans(n_clusters=n_clusters, n_init=n_init_value, random_state=42)
        kmeans.fit(features_array)

        # Assign labels to the clusters
        labels = kmeans.labels_

        # Save the trained k-means model

    else:
        print(f"Error loading image: {image_path}")

joblib.dump(kmeans, 'parking_spot_kmeans.pkl')
