#!/usr/bin/env python3
"""
Test script for SpeechPrompt v2 FastAPI service.
"""

import requests
import sys
import time
from pathlib import Path

BASE_URL = "http://localhost:8013"  # Correct port for SpeechPrompt


def test_root():
    """Test the root endpoint."""
    print("\n=== Testing GET / ===")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200


def test_status():
    """Test the status endpoint."""
    print("\n=== Testing GET /status ===")
    response = requests.get(f"{BASE_URL}/status")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Model loaded: {data.get('model_loaded')}")
    print(f"Device: {data.get('device')}")
    if data.get('vram'):
        print(f"VRAM allocated: {data['vram'].get('allocated_mb', 0):.2f} MB")
    return response.status_code == 200


def test_toggle(mode: int):
    """Test the toggle endpoint."""
    print(f"\n=== Testing POST /toggle (mode={mode}) ===")
    response = requests.post(f"{BASE_URL}/toggle", data={"mode": mode})
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {response.json()}")
    else:
        print(f"Error: {response.text}")
    return response.status_code == 200


def test_detect(audio_path: str):
    """Test the detect endpoint."""
    print(f"\n=== Testing POST /detect ===")
    print(f"Audio file: {audio_path}")
    
    if not Path(audio_path).exists():
        print(f"Error: Audio file not found: {audio_path}")
        return False
    
    start_time = time.time()
    with open(audio_path, "rb") as f:
        files = {"audio": (Path(audio_path).name, f, "audio/flac")}
        response = requests.post(f"{BASE_URL}/detect", files=files)
    
    elapsed_time = time.time() - start_time
    print(f"Status: {response.status_code}")
    print(f"Request time: {elapsed_time:.2f} seconds")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Label: {data.get('label')}")
        print(f"Confidence: {data.get('confidence', 0):.2f}%")
        print(f"Raw prediction: {data.get('raw_prediction')}")
        return True
    else:
        print(f"Error: {response.text}")
        return False


def test_detect_without_toggle(audio_path: str):
    """Test the detect endpoint without calling toggle first (auto-load test)."""
    print(f"\n=== Testing POST /detect (auto-load) ===")
    print(f"Audio file: {audio_path}")
    
    if not Path(audio_path).exists():
        print(f"Error: Audio file not found: {audio_path}")
        return False
    
    start_time = time.time()
    with open(audio_path, "rb") as f:
        files = {"audio": (Path(audio_path).name, f, "audio/flac")}
        response = requests.post(f"{BASE_URL}/detect", files=files)
    
    elapsed_time = time.time() - start_time
    print(f"Status: {response.status_code}")
    print(f"Request time (including auto-load): {elapsed_time:.2f} seconds")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Label: {data.get('label')}")
        print(f"Confidence: {data.get('confidence', 0):.2f}%")
        return True
    else:
        print(f"Error: {response.text}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("SpeechPrompt v2 FastAPI Test Suite")
    print("=" * 60)
    
    # Test basic endpoints
    test_root()
    test_status()
    
    # Test detection if audio file is provided
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        
        # Test auto-load functionality (simulates frontend behavior)
        print("\n" + "=" * 60)
        print("Testing auto-load (like frontend)...")
        print("=" * 60)
        test_detect_without_toggle(audio_path)
        
        # Check status again
        test_status()
        
        # Test normal flow with explicit toggle
        print("\n" + "=" * 60)
        print("Testing normal flow...")
        print("=" * 60)
        test_toggle(mode=0)  # Release first
        time.sleep(1)
        test_toggle(mode=1)  # Load model
        time.sleep(3)  # Wait for model to load
        test_detect(audio_path)
        
        # Toggle model off
        print("\n" + "=" * 60)
        print("Releasing model...")
        print("=" * 60)
        test_toggle(mode=0)
    else:
        print("\n" + "=" * 60)
        print("Skipping detection test (no audio file provided)")
        print("Usage: python test_speechprompt_api.py <path_to_audio_file>")
        print("=" * 60)
    
    # Final status check
    test_status()
    
    print("\n" + "=" * 60)
    print("Tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to the API server.")
        print("Make sure the server is running at", BASE_URL)
        print("\nStart the server with:")
        print("  uv run uvicorn speechprompt_fastapi:app --host 0.0.0.0 --port 8012")
        print("or")
        print("  uv run python speechprompt_fastapi.py")
        sys.exit(1)