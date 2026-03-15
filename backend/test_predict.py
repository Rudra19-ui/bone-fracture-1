import os
import sys

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fracture.predictions_engine import predict

img_path = os.path.join("media", "uploads", "WhatsApp_Image_2026-02-27_at_10.27.52_PM_1.jpeg")

if os.path.exists(img_path):
    print(f"Testing prediction on: {img_path}")
    result = predict(img_path, model="Wrist") # It was a wrist/forearm image
    print(f"Result: {result}")
else:
    print(f"Image not found: {img_path}")
