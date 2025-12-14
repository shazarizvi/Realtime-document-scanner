import cv2
import numpy as np
import imutils
import argparse
from imutils.perspective import four_point_transform


def connect_to_phone(ip):
    urls = [
        f"http://{ip}:8080/video",
        f"http://{ip}:8080/shot.jpg",
        f"http://{ip}:8080/stream.mjpg"
    ]

    print("\nTrying to connect to phone camera...\n")

    for url in urls:
        print("Testing:", url)
        cap = cv2.VideoCapture(url)
        ret, _ = cap.read()
        cap.release()

        if ret:
            print("CONNECTED using:", url, "\n")
            return url

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Document Scanner using Classical Computer Vision"
    )

    parser.add_argument(
        "--ip",
        required=True,
        help="IP address of phone running IP Webcam app"
    )

    parser.add_argument(
        "--output",
        default="scanned_phone_doc.jpg",
        help="Output filename for scanned document"
    )

    parser.add_argument(
        "--show-outline",
        action="store_true",
        help="Display detected document contour"
    )

    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable live camera preview"
    )

    args = parser.parse_args()

    cam_url = connect_to_phone(args.ip)

    if cam_url is None:
        print("\nCould NOT connect to camera. Check:")
        print("- Phone & laptop must be on SAME WiFi")
        print("- IP Webcam app must be running")
        print("- IP address must be correct")
        exit()

    cap = cv2.VideoCapture(cam_url)
    print("Press SPACE to capture the document.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lost connection!")
            exit()

        if not args.no_preview:
            cv2.imshow("Phone Camera Preview", frame)

        key = cv2.waitKey(1)
        if key == 32:  # SPACE
            image = frame.copy()
            orig = image.copy()
            print("📸 Captured!")
            break

    cap.release()
    cv2.destroyAllWindows()

    # -------- Document Scanning Pipeline --------

    ratio = image.shape[0] / 500.0
    image_small = imutils.resize(image, height=500)

    gray = cv2.cvtColor(image_small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 200)

    cnts = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    docCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            docCnt = approx
            break

    if docCnt is None:
        print("Could not detect document edges.")
        exit()

    if args.show_outline:
        outline = ima