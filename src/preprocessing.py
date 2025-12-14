# src/preprocessing.py

import cv2

def preprocess_image(image, height=500):
    """
    Resize image, convert to grayscale, and apply Gaussian blur.
    """
    ratio = image.shape[0] / height
    resized = cv2.resize(image, None, fx=1, fy=1)
    resized = cv2.resize(image, (int(image.shape[1] / ratio), height))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    return resized, blurred, ratio
