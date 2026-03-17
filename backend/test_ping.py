#!/usr/bin/env python3
"""
Test simple ping to see if server responds
"""
import requests

def test_ping():
    """Test if server responds to simple request"""
    
    try:
        response = requests.get("http://localhost:8001/api/ping", timeout=10)
        print(f"Ping Status: {response.status_code}")
        print(f"Ping Response: {response.text}")
    except Exception as e:
        print(f"Ping Error: {e}")
        
    try:
        # Test the actual analysis endpoint with minimal data
        response = requests.get("http://localhost:8001/api/analysis?image_name=test", timeout=10)
        print(f"GET Analysis Status: {response.status_code}")
        print(f"GET Analysis Response: {response.text}")
    except Exception as e:
        print(f"GET Analysis Error: {e}")

if __name__ == "__main__":
    test_ping()
