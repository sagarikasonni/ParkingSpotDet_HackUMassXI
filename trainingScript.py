import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

# Function to load images from a directory
def load_images_from_directory(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpeg") or filename.endswith(".jpg"):
            path = os.path.join(directory, filename)
            img = cv2.imread(path)
            if img is not None:
                images.append(img)
    return images

# Function to classify parking spots
def classify_parking_spots(features, labels):
    # Check if the number of features and labels is the same
    if len(features) != len(labels):
        raise ValueError("Number of features and labels must be the same.")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(np.array(X_train).reshape(-1, 1))
    X_test_scaled = scaler.transform(np.array(X_test).reshape(-1, 1))

    # Train a simple SVM classifier
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train_scaled, y_train)

    # Predict on the test set
    y_pred = svm_classifier.predict(X_test_scaled)

    # Evaluate the model
    return svm_classifier

# Specify the directory containing your dataset
dataset_directory = 'pics'

# Load images from the directory
image_list = load_images_from_directory(dataset_directory)

# Initialize lists to store features and labels
all_features = []
all_labels = []

# Loop through each image in the dataset
for img in image_list:
    # Extract features from the image
    features = extract_features_from_image(img)

    # User input for labels
    labels = [1 if input(f"Is parking spot {i} occupied in this image? (1 for yes, 0 for no): ") == '1' else 0 for i in range(len(features))]

    # Append features and labels to the overall lists
    all_features.extend(features)
    all_labels.extend(labels)

# Train the SVM classifier using the entire dataset
svm_classifier = classify_parking_spots(all_features, all_labels)

# Save the trained model
joblib.dump(svm_classifier, 'parking_spot_classifier.pkl')
