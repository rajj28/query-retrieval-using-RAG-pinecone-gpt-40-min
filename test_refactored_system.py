#!/usr/bin/env python3
"""
Test script for the refactored system with lazy loading
"""
import requests
import json
import time

def test_refactored_system():
    """Test the refactored system with lazy loading"""
    base_url = "https://hackrx-llm-system-production-1258.up.railway.app"
    
    print("🚀 Testing Refactored System with Lazy Loading")
    print("=" * 60)
    
    # Test 1: Lightweight health check (should work immediately)
    print("\n1. Testing lightweight health check...")
    try:
        response = requests.get(f"{base_url}/api/v1/hackrx/health", timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print("✅ Lightweight health check works")
    except Exception as e:
        print(f"❌ Lightweight health check failed: {e}")
    
    # Test 2: Main functionality (should trigger lazy loading)
    print("\n2. Testing main functionality (will trigger lazy loading)...")
    url = f"{base_url}/api/v1/hackrx/hackrx/run"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": "Bearer 9983c23ad9589f18637ad0a121a05b797a1f6e62fd0ff08e30bc8aa164dd618c"
    }
    data = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?"
        ]
    }
    
    start_time = time.time()
    try:
        response = requests.post(url, headers=headers, json=data, timeout=300)
        end_time = time.time()
        
        print(f"⏱️  Response Time: {end_time - start_time:.2f} seconds")
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Main functionality works with lazy loading!")
            print(f"Response: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ Main functionality failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Refactored System Test Complete!")

if __name__ == "__main__":
    test_refactored_system()
