import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform

# ----------------------------------------------------
# 0. AUTO-DETECT PHONE CAMERA STREAM FROM COMMON URLS
# ----------------------------------------------------

base_ip = "192.168.0.105"   # ← CHANGE THIS TO YOUR IP (from IP Webcam app)

urls = [
    f"http://{base_ip}:8080/video",
    f"http://{base_ip}:8080/shot.jpg",
    f"http://{base_ip}:8080/stream.mjpg"
]

cam_url = None

print("\n🔍 Trying to connect to your phone camera...\n")
for url in urls:
    print("Testing:", url)
    cap = cv2.VideoCapture(url)
    ret, frame = cap.read()
    cap.release()
    if ret:
        cam_url = url
        print("✅ CONNECTED using:", url, "\n")
        break

if cam_url is None:
    print("\n❌ Could NOT connect to camera. Check:")
    print("- Phone & laptop must be on SAME WiFi")
    print("- IP Webcam app must be running")
    print("- IP address must match exactly")
    exit()

# ----------------------------------------------------
# 1. CAPTURE IMAGE
# ----------------------------------------------------

cap = cv2.VideoCapture(cam_url)
print("Press SPACE to capture the document.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Lost connection! Re-open IP Webcam.")
        break

    cv2.imshow("Phone Camera Preview", frame)

    key = cv2.waitKey(1)
    if key == 32:  # SPACE
        image = frame.copy()
        orig = image.copy()
        print("📸 Captured!")
        break

cap.release()
cv2.destroyAllWindows()

# ----------------------------------------------------
# 2. DOCUMENT SCANNER PIPELINE
# ----------------------------------------------------

# Resize small for faster detection
height, width = image.shape[:2]
ratio = height / 500.0
image_small = imutils.resize(image, height=500)

# Grayscale
gray = cv2.cvtColor(image_small, cv2.COLOR_BGR2GRAY)

# Blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Edges
edges = cv2.Canny(blurred, 50, 200)

# Find contours
cnts = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

# Find biggest rectangular contour
docCnt = None
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        docCnt = approx
        break

if docCnt is None:
    print("❌ Could not detect document edges.")
    exit()

# Draw outline
outline = image_small.copy()
cv2.drawContours(outline, [docCnt], -1, (0, 255, 0), 2)
cv2.imshow("Detected Document Outline", outline)
cv2.waitKey(0)

# Warp perspective to original resolution
warped = four_point_transform(orig, docCnt.reshape(4, 2) * ratio)

# Convert to grayscale
warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

# Apply adaptive threshold (clean scan look)
scanned = cv2.adaptiveThreshold(
    warped_gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    10
)

# Show final scanned document
cv2.imshow("📄 Scanned Document", imutils.resize(scanned, height=700))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save
cv2.imwrite("scanned_phone_doc.jpg", scanned)
print("\n💾 Saved as scanned_phone_doc.jpg")
