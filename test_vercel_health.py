#!/usr/bin/env python3
"""
Simple test script to verify Vercel health endpoint works
"""
import requests
import json

def test_health_endpoints():
    """Test both health endpoints"""
    base_url = "http://localhost:8000"
    
    # Test 1: Root health endpoint
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print("✅ /health endpoint works")
    except Exception as e:
        print(f"❌ /health endpoint failed: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: HackRX health endpoint
    print("Testing /api/v1/hackrx/health endpoint...")
    try:
        response = requests.get(f"{base_url}/api/v1/hackrx/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print("✅ /api/v1/hackrx/health endpoint works")
    except Exception as e:
        print(f"❌ /api/v1/hackrx/health endpoint failed: {e}")

if __name__ == "__main__":
    test_health_endpoints()
