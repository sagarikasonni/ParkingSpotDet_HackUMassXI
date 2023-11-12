from google.cloud import storage

import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

import os
import io
import time
import json

# Triggered by a change in a storage bucket
def hello_gcs(cloud_event, context):

  # Set up Google Cloud credentials and project
  os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./credentials.json"
  project_id = "emptyparkingspotdetector"

  # Set up Google Cloud Storage
  bucket_name = "parking_spot_det"
  storage_client = storage.Client(project=project_id)
  bucket = storage_client.bucket(bucket_name)

  def download_from_gcs(file_name):
    blob = bucket.blob(file_name)
    content = blob.download_as_bytes()
    return io.BytesIO(content)

  def upload_to_gcs(file_name, content):
    blob = bucket.blob(file_name)
    blob.upload_from_file(content, content_type='application/octet-stream')

  pickleFile = download_from_gcs("parking_spot_kmeans.pkl")
  curImg = download_from_gcs("curImg.jpg")
  modifiedImg, details_json = runSpotDet(curImg, pickleFile)
  upload_to_gcs("modifiedImg.jpg", modifiedImg)
  upload_to_gcs("details.json", details_json)

# Function to extract features from a given image
def extract_features_from_image(image):
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
  edges = cv2.Canny(blurred_image, 50, 150)
  contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  min_contour_area = 125
  filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
  features = [cv2.contourArea(cnt) for cnt in filtered_contours]

  return features, filtered_contours

# Function to classify parking spots as free or occupied
def classify_parking_spots(features, classifier):
  if (len(features) == 0):
    return []
  features_scaled = StandardScaler().fit_transform(np.array(features).reshape(-1, 1))
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

def runSpotDet(curImg, pickleFile):
  # Extract features and classify parking spots
  image_data = np.frombuffer(curImg.read(), dtype=np.uint8)
  input_image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)  
  image_features, filtered_contours = extract_features_from_image(input_image)

  # Run algorithm
  classifier = joblib.load(io.BytesIO(pickleFile.read()))
  predictions = classify_parking_spots(image_features, classifier)
  output_image = visualize_results(input_image.copy(), filtered_contours, predictions)

  _, modified_image_data = cv2.imencode(".jpg", output_image)
  modifiedImg = io.BytesIO(modified_image_data)

  details_data = {
    "Current System Time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
    "Number of Cars": len(predictions),
  }
  details_json = io.BytesIO(json.dumps(details_data).encode())

  return modifiedImg, details_json