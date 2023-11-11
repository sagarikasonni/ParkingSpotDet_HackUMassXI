import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

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
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")

    return svm_classifier

# User input for image path
image_path = input("Enter the path to the image: ")

# Read the input image
image = cv2.imread(image_path)

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

# Dynamic feature extraction
features = [cv2.contourArea(cnt) for cnt in filtered_contours]

# Example labels: 1 for occupied, 0 for free (label your data accordingly)
labels = [1 if input(f"Is parking spot {i} occupied? (1 for yes, 0 for no): ") == '1' else 0 for i in range(len(features))]

# Classify parking spots
classifier = classify_parking_spots(features, labels)

# Visualize the results
for i, cnt in enumerate(filtered_contours):
    x, y, w, h = cv2.boundingRect(cnt)
    status = "Occupied" if classifier.predict(np.array([cv2.contourArea(cnt)]).reshape(1, -1)) == 1 else "Free"
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, status, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the results
cv2.imshow('Parking Spots', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
