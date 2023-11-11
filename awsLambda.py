import boto3
import os
import cv2
import numpy as np

s3_client = boto3.client('s3')
sns_client = boto3.client('sns')

def lambda_handler(event, context):
    # Get the S3 bucket and object key from the event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    # Download the image from S3
    local_path = '/tmp/image.jpg'
    s3_client.download_file(bucket, key, local_path)

    # Implement your parking spot detection algorithm using OpenCV or other libraries
    is_parking_spot_empty = detect_parking_spot(local_path)

    # Send notification
    if is_parking_spot_empty:
        sns_client.publish(
            TopicArn='YOUR_SNS_TOPIC_ARN',
            Message='Parking spot is empty!'
        )
    else:
        sns_client.publish(
            TopicArn='YOUR_SNS_TOPIC_ARN',
            Message='Parking spot is occupied!'
        )

def detect_parking_spot(image_path):
    # Implement your parking spot detection algorithm using OpenCV or other libraries
    # Return True if parking spot is empty, False otherwise
    pass
