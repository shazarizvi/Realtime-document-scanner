# src/camera.py
import cv2

def connect_to_phone(ip):
    """
    Try multiple IP Webcam endpoints and return a working stream URL.
    """
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
            print("CONNECTED using:", url)
            return url

    return None
