# code to detect the empty spots using OpenCV
import cv2 
import numpy as np
# starts the class containing the main working
class SpotDet:
    # constructor 
    def __init__(self):
        self.cnt = 0
    # algorithm 
    def isEmpty(self, path):
        # reading the img returned by aws 
        img = cv2.imread(path)
        # convert image to grayscale
        grayImg = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
        # now I have to reduce noise and refine contour detection
        # this can be done using gaussian blur
        blurredImg = cv2.GaussianBlur(grayImg, (5, 5), 0)
        # assuming parking spots are white -> need to define proper lower and upper bounds 
        lowerBound = np.array([200, 200, 200], dtype=np.uint8)
        upperBound = np.array([255, 255, 255], dtype=np.uint8)
        # now create a mask for all values in this lower and upper bound range 
        mask = cv2.inRange(img, lowerBound, upperBound)
        # inorder to differentiate empty and free parking spots -> need to find contours in the masked map
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # increment the counter with the length of each counter
        self.cnt += len(contours)
        # redraw contours 
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
        # final display
        cv2.imshow('The image is', img)
        # will wait indefinitely until a key is pressed since no duration is mentioned 
        cv2.waitKey(0)
        # finally close all windows
        cv2.destroyAllWindows()
    # finally return the counter
    def retCount(self):
        return self.cnt
    # TODO: adding the result from raspberry pi to image path, basically something like a main function
