#!/usr/bin/env python3
"""
Direct test of predict function to debug issues
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fracture.predictions_engine import predict

def test_predict_directly():
    """Test the predict function directly"""
    
    # Test with an actual uploaded image
    test_image = "c:/Users/rudra/OneDrive/Desktop/Bone-Fracture-Detection-master/backend/media/uploads/WhatsApp_Image_2026-02-27_at_10.27.52_PM_1.jpeg"
    
    print("Testing bone type prediction...")
    print("=" * 50)
    
    try:
        # Test bone type detection
        bone_type = predict(test_image, model="Parts", force_fresh=True)
        print(f"Bone Type Result: {bone_type}")
        
        print("\nTesting fracture prediction...")
        # Test fracture prediction
        fracture_result = predict(test_image, model=bone_type)
        print(f"Fracture Result: {fracture_result}")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_predict_directly()
