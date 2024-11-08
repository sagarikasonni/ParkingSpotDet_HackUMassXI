import time
import os
from google.cloud import storage

# Set up Google Cloud Storage configuration
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../credentials.json"
project_id = "emptyparkingspotdetector"

# Set up Google Cloud Storage
bucket_name = "parking_spot_det"
storage_client = storage.Client(project=project_id)
bucket = storage_client.bucket(bucket_name)

def download_from_gcs(file_name, local_path):
    blob = bucket.blob(file_name)
    blob.download_to_filename(local_path)

while True:
    imgPath = "../assets/modifiedImg.jpg"
    download_from_gcs("images/curImg.jpg", imgPath)

    # detailsPath = os.path.abspath("data/details.json")
    # download_from_gcs("details.json", detailsPath)

    time.sleep(1)