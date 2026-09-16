# Face Detection using Haar Cascades with OpenCV and Matplotlib

## Aim

To write a Python program using OpenCV to perform the following image manipulations:  
i) Extract ROI from an image.  
ii) Perform face detection using Haar Cascades in static images.  
iii) Perform eye detection in images.  
iv) Perform face detection with label in real-time video from webcam.

## Software Required

- Anaconda - Python 3.7 or above  
- OpenCV library (`opencv-python`)  
- Matplotlib library (`matplotlib`)  
- Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm

### I) Load and Display Images

- Step 1: Import necessary packages: `numpy`, `cv2`, `matplotlib.pyplot`  
- Step 2: Load grayscale images using `cv2.imread()` with flag `0`  
- Step 3: Display images using `plt.imshow()` with `ccap='gray'`

### II) Load Haar Cascade Classifiers

- Step 1: Load face and eye cascade XML files 
### III) Perform Face Detection in Images

- Step 1: Define a function `detect_face()` that copies the input image  
- Step 2: Use `face_cascade.detectMultiScale()` to detect faces  
- Step 3: Draw white rectangles around detected faces with thickness 10  
- Step 4: Return the processed image with rectangles  

### IV) Perform Eye Detection in Images

- Step 1: Define a function `detect_eyes()` that copies the input image  
- Step 2: Use `eye_cascade.detectMultiScale()` to detect eyes  
- Step 3: Draw white rectangles around detected eyes with thickness 10  
- Step 4: Return the processed image with rectangles  

### V) Display Detection Results on Images

- Step 1: Call `detect_face()` or `detect_eyes()` on loaded images  
- Step 2: Use `plt.imshow()` with `ccap='gray'` to display images with detected regions highlighted  

### VI) Perform Face Detection on Real-Time Webcam Video

- Step 1: Capture video from webcam using `cv2.VideoCapture(0)`  
- Step 2: Loop to continuously read frames from webcam  
- Step 3: Apply `detect_face()` function on each frame  
- Step 4: Display the video frame with rectangles around detected faces  
- Step 5: Exit loop and close windows when ESC key (key code 27) is pressed  
- Step 6: Release video capture and destroy all OpenCV windows

  ### program
```
  import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# PART 1: ROI SEGMENTATION
# =========================

image = cv2.imread('cc.jpg')

if image is None:
    print("Error: cc.jpg not found")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')
plt.show()

# ROI
roi = image[100:420, 200:550]

mask = np.zeros_like(image)
mask[100:420, 200:550] = roi

segmented = cv2.bitwise_and(image, mask)

plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
plt.title("Segmented ROI")
plt.axis('off')
plt.show()


# =========================
# PART 2: EDGE DETECTION
# =========================

image = cv2.imread('wh.jpg')

if image is None:
    print("Error: wh.jpg not found")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

plt.imshow(edges, ccap='gray')
plt.title("Canny Edge Detection")
plt.axis('off')
plt.show()

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

result = image.copy()
          
for c in contours:
    if cv2.contourArea(c) > 50:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)

plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Contour Detection")
plt.axis('off')
plt.show()


# =========================
# PART 3: OBJECT DETECTION (SAFE VERSION)
# =========================

import cv2
import numpy as np
import matplotlib.pyplot as plt

print("Libraries imported successfully")

print("OpenCV version:", cv2.__version__)
print("DNN module available:", hasattr(cv2, "dnn"))
print("readNet available:", hasattr(cv2.dnn, "readNet"))

config_file = "deploy.prototxt"
weights = "mobilenet_iter_73000.caffemodel"

print("Config file:", config_file)
print("Weights file:", weights)

net = cv2.dnn.readNet(weights, config_file)

print("MobileNet-SSD model loaded successfully!")

class_labels = {
    0: "background",
    1: "aeroplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "diningtable",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "pottedplant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tvmonitor"
}

print("Class labels created")

image = cv2.imread("bike.jpg")

if image is None:
    print("ERROR: bike.jpg not found")
else:
    print("Image loaded successfully")
    print("Image size:", image.shape)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8, 6))
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()

blob = cv2.dnn.blobFromImage(
    image,
    scalefactor=0.007843,
    size=(300, 300),
    mean=127.5
)

print("Blob created successfully")
print("Blob shape:", blob.shape)

net.setInput(blob)

detections = net.forward()

print("Detection completed successfully")
print("Detection shape:", detections.shape)

confidence_threshold = 0.5

for i in range(detections.shape[2]):

    confidence = detections[0, 0, i, 2]

    if confidence > confidence_threshold:

        index = int(detections[0, 0, i, 1])

        label = class_labels.get(index, "Unknown")

        print(
            label,
            "Confidence:",
            round(confidence * 100, 2),
            "%"
        )

# Step 11: Draw larger bounding boxes and labels

(h, w) = image.shape[:2]

for i in range(detections.shape[2]):

    confidence = detections[0, 0, i, 2]

    if confidence > 0.5:

        index = int(detections[0, 0, i, 1])
        label = class_labels.get(index, "Unknown")

        box = detections[0, 0, i, 3:7] * np.array(
            [w, h, w, h]
        )

        startX, startY, endX, endY = box.astype("int")

        startX = max(0, startX)
        startY = max(0, startY)
        endX = min(w - 1, endX)
        endY = min(h - 1, endY)

        # Large label text
        text = "{}: {:.1f}%".format(
            label,
            confidence * 100
        )

        # Draw thick bounding box
        cv2.rectangle(
            image,
            (startX, startY),
            (endX, endY),
            (0, 255, 0),
            3
        )

        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3

        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness
        )

        # Draw filled background behind text
        cv2.rectangle(
            image,
            (startX, max(0, startY - text_height - baseline - 10)),
            (startX + text_width + 10, startY),
            (0, 255, 0),
            -1
        )

        # Draw large label
        cv2.putText(
            image,
            text,
            (startX + 5, startY - 5),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA
        )

print("Large labels and bounding boxes drawn successfully")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 7))
plt.imshow(image_rgb)
plt.title("Object Detection using MobileNet-SSD")
plt.axis("off")
plt.show()
```
```
```
  ### Output:

- Original image:
<img width="552" height="365" alt="download" src="https://github.com/user-attachments/assets/be36c8a3-fc75-4fdf-b8e1-64eaa36b2d62" />

-Segmented ROI:
<img width="552" height="365" alt="download" src="https://github.com/user-attachments/assets/f5119d99-9999-44f4-96c2-6627acd6775e" />
- Original image:
<img width="515" height="369" alt="download" src="https://github.com/user-attachments/assets/8e1758d7-a9fe-4cfc-8c43-9b3866d82bf0" />

-Canny Edge Detection:
<img width="515" height="369" alt="download" src="https://github.com/user-attachments/assets/5e33e971-0e10-409d-90c7-87457768553f" />

- Handwriting Detection:
 <img width="515" height="369" alt="download" src="https://github.com/user-attachments/assets/b7acfba3-0757-482f-a01f-f92f5ac54810" />

- Original image:
  <img width="279" height="502" alt="download" src="https://github.com/user-attachments/assets/e5542900-a7ce-4775-af8e-3307bc4979e0" />

- Object Detection using MobileNet-SSD:
<img width="340" height="579" alt="download" src="https://github.com/user-attachments/assets/243bc71a-51bd-4398-b584-2b1b3756dca6" />

### RESULT

Thus to write a Python program using OpenCV to perform the following image manipulations was verified successfully.

