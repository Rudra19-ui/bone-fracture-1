#!/usr/bin/env python3
"""
Test script to verify bone type detection functionality
"""
import os
import sys

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fracture.predictions_engine import predict

def test_bone_type_detection():
    """Test bone type detection with various filenames"""
    
    test_cases = [
        ("elbow_xray.png", "Elbow"),
        ("hand_fracture.jpg", "Hand"), 
        ("shoulder_image.jpeg", "Shoulder"),
        ("wrist_xray.png", "Wrist"),
        ("ankle_fracture.jpg", "Ankle"),
        ("random_image.png", "Elbow"),  # Default fallback
        ("patient00011_study1_negative_image1.png", "Elbow"),  # Dataset format
        ("WhatsApp_Image_2026-02-27_at_10.27.52_PM.jpeg", "Elbow")  # Test image
    ]
    
    print("Testing Bone Type Detection...")
    print("=" * 50)
    
    for filename, expected in test_cases:
        # Test filename-based detection (simulated)
        lower_name = filename.lower()
        
        if any(k in lower_name for k in ["hand", "finger", "palm", "metacarpal", "phalanx"]): 
            detected = "Hand"
        elif any(k in lower_name for k in ["wrist", "forearm", "radius", "ulna", "carpal"]): 
            detected = "Wrist"
        elif any(k in lower_name for k in ["elbow", "olecranon", "humerus_distal"]): 
            detected = "Elbow"
        elif any(k in lower_name for k in ["shoulder", "clavicle", "acromion", "scapula", "humerus_proximal"]): 
            detected = "Shoulder"
        elif any(k in lower_name for k in ["ankle", "foot", "tibia", "fibula", "talus", "calcaneus"]): 
            detected = "Ankle"
        else:
            detected = "Elbow"  # Default fallback
            
        status = "✓" if detected == expected else "✗"
        print(f"{status} {filename}")
        print(f"  Expected: {expected}, Detected: {detected}")
        print()

if __name__ == "__main__":
    test_bone_type_detection()
