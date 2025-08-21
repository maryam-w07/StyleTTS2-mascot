#!/usr/bin/env python3
"""
Complete example script for testing the StyleTTS2 FastAPI wrapper.
This script demonstrates how to use the API programmatically.
"""

import requests
import json
import time
import numpy as np
import soundfile as sf
import os
import tempfile

def create_test_audio(filename="test_audio.wav", duration=2.0, sample_rate=24000):
    """Create a simple test audio file"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Create a simple melody with multiple frequencies
    audio = 0.3 * (np.sin(2 * np.pi * 440 * t) + 
                   0.5 * np.sin(2 * np.pi * 880 * t) + 
                   0.3 * np.sin(2 * np.pi * 220 * t))
    
    sf.write(filename, audio, sample_rate)
    return filename

def test_api_health(base_url="http://localhost:8000"):
    """Test the API health endpoints"""
    print("Testing API health...")
    
    # Test root endpoint
    try:
        response = requests.get(f"{base_url}/")
        print(f"✓ Root endpoint: {response.json()}")
    except Exception as e:
        print(f"✗ Root endpoint failed: {e}")
        return False
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        health_data = response.json()
        print(f"✓ Health endpoint: {health_data}")
        return health_data.get("status") == "healthy"
    except Exception as e:
        print(f"✗ Health endpoint failed: {e}")
        return False

def test_viseme_generation(base_url="http://localhost:8000", audio_file="test_audio.wav"):
    """Test the viseme generation endpoints"""
    print(f"\nTesting viseme generation with {audio_file}...")
    
    test_texts = [
        "Hello world",
        "This is a test",
        "The quick brown fox jumps over the lazy dog",
        "How are you today",
        "Machine learning is fascinating"
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\nTest {i+1}: '{text}'")
        
        try:
            # Test full response endpoint
            with open(audio_file, 'rb') as audio:
                response = requests.post(
                    f"{base_url}/generate_visemes",
                    files={'audio_file': audio},
                    data={'text': text}
                )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Full response - Status: {result['status']}")
                print(f"  Inference mode: {result['inference_mode']}")
                print(f"  Visemes count: {len(result['visemes'])}")
                print(f"  First few visemes: {result['visemes'][:3]}")
            else:
                print(f"✗ Full response failed: {response.status_code} - {response.text}")
                continue
                
        except Exception as e:
            print(f"✗ Full response error: {e}")
            continue
        
        try:
            # Test JSON-only endpoint
            with open(audio_file, 'rb') as audio:
                response = requests.post(
                    f"{base_url}/generate_visemes_json",
                    files={'audio_file': audio},
                    data={'text': text}
                )
            
            if response.status_code == 200:
                visemes = response.json()
                print(f"✓ JSON-only response - Visemes count: {len(visemes)}")
                if visemes:
                    print(f"  Duration: {visemes[-1]['offset']:.2f}s")
            else:
                print(f"✗ JSON-only response failed: {response.status_code}")
                
        except Exception as e:
            print(f"✗ JSON-only response error: {e}")

def test_error_handling(base_url="http://localhost:8000"):
    """Test API error handling"""
    print("\nTesting error handling...")
    
    # Test with invalid file format
    try:
        temp_file = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        temp_file.write(b"This is not an audio file")
        temp_file.close()
        
        with open(temp_file.name, 'rb') as file:
            response = requests.post(
                f"{base_url}/generate_visemes",
                files={'audio_file': file},
                data={'text': 'test text'}
            )
        
        if response.status_code == 400:
            print("✓ Invalid file format correctly rejected")
        else:
            print(f"✗ Invalid file format not handled: {response.status_code}")
            
        os.unlink(temp_file.name)
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
    
    # Test with empty text
    try:
        audio_file = create_test_audio("temp_test.wav", duration=1.0)
        
        with open(audio_file, 'rb') as file:
            response = requests.post(
                f"{base_url}/generate_visemes",
                files={'audio_file': file},
                data={'text': ''}
            )
        
        if response.status_code == 400:
            print("✓ Empty text correctly rejected")
        else:
            print(f"✗ Empty text not handled: {response.status_code}")
            
        os.unlink(audio_file)
    except Exception as e:
        print(f"✗ Empty text test failed: {e}")

def test_different_audio_formats(base_url="http://localhost:8000"):
    """Test different audio formats"""
    print("\nTesting different audio formats...")
    
    # Create test files in different formats
    test_files = []
    
    # WAV file
    wav_file = create_test_audio("test.wav", duration=1.5)
    test_files.append(("WAV", wav_file))
    
    # For MP3, we would need additional dependencies, so we'll skip for now
    # but the API should handle it
    
    test_text = "Testing audio formats"
    
    for format_name, file_path in test_files:
        try:
            with open(file_path, 'rb') as audio:
                response = requests.post(
                    f"{base_url}/generate_visemes",
                    files={'audio_file': audio},
                    data={'text': test_text}
                )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ {format_name} format works - {len(result['visemes'])} visemes")
            else:
                print(f"✗ {format_name} format failed: {response.status_code}")
                
            os.unlink(file_path)
        except Exception as e:
            print(f"✗ {format_name} format error: {e}")

def benchmark_api(base_url="http://localhost:8000", num_requests=5):
    """Benchmark API performance"""
    print(f"\nBenchmarking API with {num_requests} requests...")
    
    audio_file = create_test_audio("benchmark_audio.wav", duration=3.0)
    test_text = "This is a benchmark test for the API performance"
    
    times = []
    
    for i in range(num_requests):
        start_time = time.time()
        
        try:
            with open(audio_file, 'rb') as audio:
                response = requests.post(
                    f"{base_url}/generate_visemes_json",
                    files={'audio_file': audio},
                    data={'text': test_text}
                )
            
            if response.status_code == 200:
                elapsed = time.time() - start_time
                times.append(elapsed)
                print(f"  Request {i+1}: {elapsed:.2f}s")
            else:
                print(f"  Request {i+1}: Failed ({response.status_code})")
                
        except Exception as e:
            print(f"  Request {i+1}: Error - {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        print(f"\n✓ Benchmark results:")
        print(f"  Average: {avg_time:.2f}s")
        print(f"  Min: {min_time:.2f}s")
        print(f"  Max: {max_time:.2f}s")
    
    os.unlink(audio_file)

def main():
    """Main test function"""
    print("StyleTTS2 FastAPI Test Suite")
    print("=" * 40)
    
    base_url = "http://localhost:8000"
    
    # Check if API is running
    if not test_api_health(base_url):
        print("\n✗ API is not running or not healthy!")
        print("Please start the API server with: python api_simple.py")
        return
    
    # Create test audio file
    print(f"\nCreating test audio file...")
    audio_file = create_test_audio("test_audio.wav", duration=2.0)
    print(f"✓ Created {audio_file}")
    
    # Run tests
    test_viseme_generation(base_url, audio_file)
    test_error_handling(base_url)
    test_different_audio_formats(base_url)
    benchmark_api(base_url)
    
    # Cleanup
    if os.path.exists(audio_file):
        os.unlink(audio_file)
    
    print("\n" + "=" * 40)
    print("Test suite completed!")

if __name__ == "__main__":
    main()