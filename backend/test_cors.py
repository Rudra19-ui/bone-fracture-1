#!/usr/bin/env python3
"""
Test CORS with the frontend origin
"""
import requests

def test_cors():
    """Test CORS with frontend origin"""
    
    url = "http://localhost:8001/api/analysis"
    headers = {
        'Origin': 'http://localhost:3002',
        'Content-Type': 'application/json'
    }
    
    try:
        # Test OPTIONS request (CORS preflight)
        response = requests.options(url, headers=headers, timeout=10)
        print(f"OPTIONS Status: {response.status_code}")
        print(f"OPTIONS Headers: {dict(response.headers)}")
        
        # Check for CORS headers
        cors_headers = {
            'Access-Control-Allow-Origin': response.headers.get('Access-Control-Allow-Origin'),
            'Access-Control-Allow-Methods': response.headers.get('Access-Control-Allow-Methods'),
            'Access-Control-Allow-Headers': response.headers.get('Access-Control-Allow-Headers')
        }
        print(f"CORS Headers: {cors_headers}")
        
    except Exception as e:
        print(f"CORS Test Error: {e}")

if __name__ == "__main__":
    test_cors()
