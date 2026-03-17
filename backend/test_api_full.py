#!/usr/bin/env python3
"""
Test the API endpoint directly
"""
import requests
import json

def test_api_endpoint():
    """Test the full API endpoint"""
    
    url = "http://localhost:8001/api/analysis"
    
    # Test with the same image
    image_path = "c:/Users/rudra/OneDrive/Desktop/Bone-Fracture-Detection-master/backend/media/uploads/WhatsApp_Image_2026-02-27_at_10.27.52_PM_1.jpeg"
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {
                'image_name': 'WhatsApp_Image_2026-02-27_at_10.27.52_PM_1.jpeg',
                'user_name': 'Test User',
                'user_type': 'doctor'
            }
            
            print("Sending request to API...")
            response = requests.post(url, files=files, data=data, timeout=60)
            
            print(f"Status Code: {response.status_code}")
            print(f"Response Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                result = response.json()
                print("Response JSON:")
                print(json.dumps(result, indent=2))
            else:
                print(f"Error Response: {response.text}")
                
    except requests.exceptions.Timeout:
        print("REQUEST TIMEOUT - Server took too long to respond")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api_endpoint()
