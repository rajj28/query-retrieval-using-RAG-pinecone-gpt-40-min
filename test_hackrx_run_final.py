#!/usr/bin/env python3
"""
Final test for the /hackrx/run endpoint with actual data
"""
import requests
import json
import time

def test_hackrx_run_final():
    """Test the main HackRX run endpoint with actual data"""
    base_url = "https://hackrx-llm-system-production-1258.up.railway.app"
    
    # API endpoint
    url = f"{base_url}/api/v1/hackrx/hackrx/run"
    
    # Headers
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": "Bearer 9983c23ad9589f18637ad0a121a05b797a1f6e62fd0ff08e30bc8aa164dd618c"
    }
    
    # Request payload (actual data from user)
    data = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?",
            "What is the waiting period for cataract surgery?",
            "Are the medical expenses for an organ donor covered under this policy?",
            "What is the No Claim Discount (NCD) offered in this policy?",
            "Is there a benefit for preventive health check-ups?",
            "How does the policy define a 'Hospital'?",
            "What is the extent of coverage for AYUSH treatments?",
            "Are there any sub-limits on room rent and ICU charges for Plan A?"
        ]
    }
    
    print("🚀 Testing HackRX Run Endpoint - FINAL TEST")
    print("=" * 60)
    print(f"URL: {url}")
    print(f"Questions: {len(data['questions'])}")
    print(f"Document: {data['documents'][:50]}...")
    print("\n⏳ Sending request...")
    
    start_time = time.time()
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=300)  # 5 minute timeout
        end_time = time.time()
        
        print(f"⏱️  Response Time: {end_time - start_time:.2f} seconds")
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! Response received:")
            print(json.dumps(result, indent=2))
            
            # Verify response format
            if "answers" in result and isinstance(result["answers"], list):
                print(f"\n✅ Response format correct: {len(result['answers'])} answers")
                print("\n📋 Answers Summary:")
                for i, answer in enumerate(result['answers'], 1):
                    print(f"{i}. {answer[:100]}...")
                
                # Check if answers are meaningful
                meaningful_answers = [ans for ans in result['answers'] if len(ans) > 20 and not ans.startswith('Error')]
                print(f"\n✅ Meaningful answers: {len(meaningful_answers)}/{len(result['answers'])}")
                
            else:
                print("❌ Response format does not match expected structure")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out (5 minutes)")
    except requests.exceptions.ConnectionError:
        print("❌ Connection error")
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Final HackRX Run Test Complete!")

if __name__ == "__main__":
    test_hackrx_run_final()
