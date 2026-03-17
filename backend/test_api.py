#!/usr/bin/env python3
"""
Test the enhanced bone type detection with a sample image
"""
import requests
import json

def test_backend_api():
    """Test the backend API with a sample image"""
    
    # Test with one of the uploaded images
    image_path = "c:/Users/rudra/OneDrive/Desktop/Bone-Fracture-Detection-master/backend/media/uploads/WhatsApp_Image_2026-02-27_at_10.27.52_PM_1.jpeg"
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {
                'image_name': 'WhatsApp_Image_2026-02-27_at_10.27.52_PM_1.jpeg',
                'user_name': 'Test User',
                'user_type': 'doctor'
            }
            
            response = requests.post('http://localhost:8000/api/analysis', files=files, data=data)
            
            print("API Response Status:", response.status_code)
            print("API Response:")
            print(json.dumps(response.json(), indent=2))
            
    except Exception as e:
        print(f"Error testing API: {e}")

if __name__ == "__main__":
    test_backend_api()
