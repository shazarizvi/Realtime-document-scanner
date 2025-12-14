## Overview

This project implements a real-time document scanner using **classical computer vision techniques** in Python and OpenCV. Given an image containing a document, the system automatically detects document boundaries and applies a perspective transformation to produce a clean, top-down scanned output.

The primary goal of this project is **educational and exploratory**: to deeply understand the end-to-end computer vision pipeline behind document scanning applications without relying on end-to-end machine learning models. Emphasis is placed on modular design, interpretability, and experimentation.

This repository is designed to be **extensible and research-oriented**, making it suitable as a foundation for further improvements such as robustness enhancements, OCR integration, and hybrid classical–ML approaches.

---
## Motivation

This project was built to strengthen foundational computer vision skills and serve as a base for further experimentation and open-source contributions.

---

## Key Features

- Automatic document boundary detection
- Edge detection and contour analysis using OpenCV
- Perspective transformation for flattened, scan-like output
- Lightweight and fast pipeline suitable for near real-time usage
- Modular, readable codebase focused on learning and experimentation

---

## Tech Stack

- **Language:** Python
- **Computer Vision:** OpenCV
- **Numerical Computing:** NumPy
- **Version Control:** Git, GitHub

---

## Pipeline Overview

The document scanner follows a sequential image-processing pipeline, where each stage prepares the image for more reliable detection downstream:

1. **Image Resizing**
    
    The input image is resized to a fixed height to reduce computation time and normalize contour area thresholds across varying image resolutions.
    
2. **Grayscale Conversion**
    
    The resized image is converted to grayscale to reduce dimensionality and emphasize structural features over color information.
    
3. **Noise Reduction**
    
    Gaussian blurring is applied to suppress high-frequency noise and minor texture details that may interfere with edge detection.
    
4. **Edge Detection**
    
    Canny edge detection is used to highlight strong intensity gradients corresponding to document boundaries.
    
5. **Contour Extraction and Filtering**
    
    Contours are extracted and approximated to polygons. The largest four-point contour is assumed to correspond to the document.
    
6. **Perspective Transformation**
    
    A homography-based perspective transform is applied to the detected corners to generate a top-down, scanned view of the document.
    

---

## Project Structure

```
Document-Scanner-using-classical-computer-vision/
├── src/
│   ├── camera.py 
│   ├── preprocessing.py        # Grayscale conversion and blurring
│   ├── edge_detection.py       # Edge detection strategies
│   ├── contour_detection.py    # Contour extraction and filtering
│   ├── perspective.py          # Perspective transformation utilities
│   └── scanner.py              # Pipeline orchestration
├── outputs/                    # Saved scanned outputs
├── requirements.txt
└── README.md

```

---

## Usage

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt

```

### Run using Phone Camera (IP Webcam)

Ensure your phone and laptop are on the same WiFi network and the **IP Webcam** app is running on your phone.

```bash
python src/scanner.py --ip <PHONE_IP_ADDRESS>

```

Example:

```bash
python src/scanner.py --ip 192.168.1.5

```

### Optional Flags

```bash
--output <filename>        Save scanned image with a custom name
--show-outline             Display detected document contour
--no-preview               Disable live camera preview

```

Example:

```bash
python src/scanner.py --ip 192.168.1.5 --show-outline --output scanned_doc.jpg

```

The scanned output image is saved to the specified output path.

---

## Experimental Observations

Multiple preprocessing and edge detection strategies were evaluated to study their impact on document boundary detection reliability under varying conditions.

| Scenario | Technique Varied | Observation | Insight |
| --- | --- | --- | --- |
| Good lighting | Canny edge detection | Clean edges and accurate quadrilateral detection | Default pipeline performs reliably |
| Low contrast | Canny edge detection | Fragmented edges and missed document corners | Highly sensitive to contrast |
| Low contrast | Adaptive thresholding | Improved boundary continuity and contour detection | Better suited for uneven illumination |
| Cluttered background | Increased Gaussian blur | Reduced false-positive contours | Noise suppression is critical |

These experiments indicate that **preprocessing choices significantly influence downstream contour detection performance**, often more than the edge detection method itself. Gradient-based detectors such as Canny are particularly sensitive to contrast variations, while local thresholding methods demonstrate improved robustness under uneven lighting.

### Failure Cases

- Strong shadows resulted in merged contours
- Reflections on glossy paper confused edge detection
- Non-rectangular documents violated the four-corner assumption

These observations motivate future improvements focused on adaptive preprocessing, shadow handling, and more flexible document shape modeling.

---

## Sample Results

*(Add* 

- *Original input image*
- *Edge-detected image*
- *Detected document contour*
- *Final perspective-corrected output*

*later)*

---

## Design Decisions

- **Classical CV over ML:** Chosen for interpretability, low latency, and to deeply understand fundamental vision techniques.
- **Largest quadrilateral assumption:** Simplifies detection while remaining effective under controlled conditions.
- **Fixed image resizing:** Stabilizes contour filtering heuristics across inputs.
- **Modular pipeline design:** Enables easy experimentation, debugging, and future extension.

---

## Limitations

- Performance degrades under strong shadows, reflections, or very low-contrast documents.
- Assumes a single prominent document in the frame.
- Automatic corner detection may fail in highly cluttered scenes.

---

## Future Work

- Investigate adaptive and local thresholding methods for improved robustness
- Add confidence scoring for detected corners
- Integrate OCR for text extraction and document digitization
- Support batch processing of image directories
- Add an interactive fallback mode for manual corner adjustment
- Explore hybrid classical + lightweight ML approaches for boundary detection

---

## What I Learned

- Preprocessing choices often have a larger impact on performance than downstream algorithms
- Perspective transformation is reliable only when corner detection is precise
- Modular pipeline design significantly simplifies debugging and experimentation
- Classical computer vision pipelines work well under controlled conditions but require careful tuning for real-world variability

---

## Author

**Shaza Rizvi**

GitHub: https://github.com/shazarizvi

LinkedIn: https://www.linkedin.com/in/shaza-rizvi-692833254/

---

## License

This project is open-source and intended for educational and research purposes.