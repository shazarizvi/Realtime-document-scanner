# src/perspective.py
import cv2
from imutils.perspective import four_point_transform

def warp_perspective(orig, docCnt, ratio):
    """
    Apply perspective transform to get top-down view.
    """
    warped = four_point_transform(orig, docCnt.reshape(4, 2) * ratio)

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    scanned = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        10
    )

    return scanned
