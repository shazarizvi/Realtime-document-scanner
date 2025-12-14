# src/edge_detection.py
import cv2

def detect_edges(blurred):
    """
    Apply Canny edge detection.
    """
    edges = cv2.Canny(blurred, 50, 200)
    return edges
