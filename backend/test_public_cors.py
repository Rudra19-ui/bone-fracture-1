#!/usr/bin/env python3
"""
Test public CORS access with any origin
"""
import requests

def test_public_cors():
    """Test CORS with different origins"""
    
    url = "http://localhost:8001/api/analysis"
    
    # Test with different origins
    test_origins = [
        'http://example.com',
        'https://any-domain.com',
        'http://192.168.1.100:3000',
        'file://'
    ]
    
    for origin in test_origins:
        headers = {
            'Origin': origin,
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.options(url, headers=headers, timeout=10)
            cors_origin = response.headers.get('access-control-allow-origin')
            print(f"Origin: {origin}")
            print(f"Allowed: {cors_origin}")
            print("---")
            
        except Exception as e:
            print(f"Error testing {origin}: {e}")

if __name__ == "__main__":
    test_public_cors()
