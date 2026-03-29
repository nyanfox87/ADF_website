"""
Test script for Whisper FastAPI service
Tests all endpoints: /status, /toggle, and /asr
"""

import requests
import os
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8021"
TEST_AUDIO_PATH = "poisoned_pen_01_f000239.wav"  # Path to a test audio file
TEST_AUDIO_PATH2 = "vo_XGLQ202_13_avin_09.wav"

def test_status():
    """Test the /status endpoint"""
    print("\n=== Testing /status endpoint ===")
    response = requests.get(f"{BASE_URL}/status")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_toggle_load():
    """Test loading the model via /toggle endpoint"""
    print("\n=== Testing /toggle endpoint (Load Model) ===")
    response = requests.post(f"{BASE_URL}/toggle")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_toggle_unload():
    """Test unloading the model via /toggle endpoint"""
    print("\n=== Testing /toggle endpoint (Unload Model) ===")
    response = requests.post(f"{BASE_URL}/toggle")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_asr_without_model():
    """Test /asr endpoint when model is not loaded"""
    print("\n=== Testing /asr endpoint (Model Not Loaded) ===")
    
    # Create a dummy file for testing
    test_file = {"file": ("test.wav", b"dummy data", "audio/wav")}
    response = requests.post(f"{BASE_URL}/asr", files=test_file)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 503

def test_asr_with_audio(audio_path):
    """Test /asr endpoint with actual audio file"""
    print("\n=== Testing /asr endpoint (With Audio) ===")
    
    if not os.path.exists(audio_path):
        print(f"⚠️  Audio file not found: {audio_path}")
        print("Please provide a valid audio file path to test transcription")
        return None
    
    with open(audio_path, "rb") as f:
        files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
        response = requests.post(f"{BASE_URL}/asr", files=files)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n📝 Transcription: {result.get('text')}")
        print(f"🌍 Language: {result.get('language')}")
    
    return response.status_code == 200

def run_all_tests():
    """Run all tests in sequence"""
    print("=" * 60)
    print("🧪 Starting Whisper FastAPI Tests")
    print("=" * 60)
    
    results = {}
    
    try:
        # Test 1: Check initial status
        results["status"] = test_status()
        
        # Test 2: Test ASR without model loaded (should fail)
        results["asr_no_model"] = test_asr_without_model()
        
        # Test 3: Load model
        results["toggle_load"] = test_toggle_load()
        
        # Test 4: Check status after loading
        results["status_after_load"] = test_status()
        
        # Test 5: Test ASR with model loaded
        if os.path.exists(TEST_AUDIO_PATH):
            results["asr_with_audio"] = test_asr_with_audio(TEST_AUDIO_PATH)
        else:
            print(f"\n⚠️  Skipping audio transcription test: {TEST_AUDIO_PATH} not found")
            print("To test transcription, provide an audio file path")

        # Test 6: Test ASR with second audio file if exists
        if os.path.exists(TEST_AUDIO_PATH2):
            results["asr_with_audio_2"] = test_asr_with_audio(TEST_AUDIO_PATH2)
        else:
            print(f"\n⚠️  Skipping audio transcription test: {TEST_AUDIO_PATH2} not found")
            print("To test transcription, provide an audio file path")
        
        # Test 7: Unload model
        results["toggle_unload"] = test_toggle_unload()
        
        # Test 8: Check status after unloading
        results["status_after_unload"] = test_status()
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Connection Error: Unable to connect to the API")
        print(f"Make sure the server is running at {BASE_URL}")
        return
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v is True)
    total = len([v for v in results.values() if v is not None])
    
    for test_name, result in results.items():
        if result is None:
            status = "⊘ SKIPPED"
        elif result:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\n🎯 Total: {passed}/{total} tests passed")
    print("=" * 60)

if __name__ == "__main__":
    import sys
    
    # Allow custom audio file path as command line argument
    if len(sys.argv) > 1:
        TEST_AUDIO_PATH = sys.argv[1]
        print(f"Using audio file: {TEST_AUDIO_PATH}")
    
    run_all_tests()
