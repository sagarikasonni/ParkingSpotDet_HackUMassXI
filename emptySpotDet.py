import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
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

    return features, filtered_contours

# Function to classify parking spots as free or occupied
def classify_parking_spots(features, classifier):
    # Feature scaling
    features_scaled = StandardScaler().fit_transform(np.array(features).reshape(-1, 1))

    # Classify parking spots using the pre-trained SVM model
    predictions = classifier.predict(features_scaled)

    return predictions

# Function to visualize the results
def visualize_results(image, filtered_contours, predictions):
    for i, cnt in enumerate(filtered_contours):
        x, y, w, h = cv2.boundingRect(cnt)
        status = "Occupied" if predictions[i] == 0 else "Free"
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, status, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

# Load the pre-trained SVM model
classifier = joblib.load('parking_spot_kmeans.pkl')  # Replace with your actual filename

# Read the input image
input_image = cv2.imread('test1.jpeg')

# Extract features from the image
image_features, filtered_contours = extract_features_from_image(input_image)

# Classify parking spots
predictions = classify_parking_spots(image_features, classifier)

# Visualize the results
output_image = visualize_results(input_image.copy(), filtered_contours, predictions)

# Display the results
cv2.imshow('Parking Spots', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
