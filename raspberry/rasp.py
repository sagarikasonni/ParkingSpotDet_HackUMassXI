import os
import time
from picamera import PiCamera
from google.cloud import storage

# Set up Google Cloud credentials and project
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./credentials.json"
project_id = "emptyparkingspotdetector"

# Set up Google Cloud Storage
bucket_name = "parking_spot_det"
storage_client = storage.Client(project=project_id)
bucket = storage_client.bucket(bucket_name)

# Set up the camera
camera = PiCamera()

def capture_image():
    image_path = './curImg.jpg'
    camera.capture(image_path)
    return image_path

def upload_to_gcs(file_path):
    file_name = os.path.basename(file_path)
    blob = bucket.blob(f'images/{file_name}')
    blob.upload_from_filename(file_path)

# Main loop
while True:
    image_path = capture_image()
    upload_to_gcs(image_path)
    time.sleep(1)
    
