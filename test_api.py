#!/usr/bin/env python3
"""
Minimal test script to check imports and basic functionality
"""

import os
import sys
import tempfile

def test_basic_imports():
    """Test basic imports"""
    try:
        import yaml
        print("✓ yaml import successful")
        
        import munch
        print("✓ munch import successful")
        
        import json
        print("✓ json import successful")
        
        import torch
        print("✓ torch import successful")
        
        import torchaudio
        print("✓ torchaudio import successful")
        
        import numpy as np
        print("✓ numpy import successful")
        
        import librosa
        print("✓ librosa import successful")
        
        import soundfile as sf
        print("✓ soundfile import successful")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_fastapi_imports():
    """Test FastAPI imports"""
    try:
        from fastapi import FastAPI, File, UploadFile, Form
        print("✓ FastAPI imports successful")
        
        import uvicorn
        print("✓ uvicorn import successful")
        
        return True
    except ImportError as e:
        print(f"✗ FastAPI import error: {e}")
        return False

def test_local_imports():
    """Test local module imports"""
    try:
        # Test individual components
        from text_utils import TextCleaner
        print("✓ TextCleaner import successful")
        
        # Test if we can load config
        if os.path.exists('config_ft.yml'):
            import yaml
            with open('config_ft.yml', 'r') as f:
                config = yaml.safe_load(f)
            print("✓ Config loading successful")
        else:
            print("✗ config_ft.yml not found")
            
        return True
    except ImportError as e:
        print(f"✗ Local import error: {e}")
        return False

def create_simple_api():
    """Create a simple test API"""
    from fastapi import FastAPI, File, UploadFile, Form
    from fastapi.responses import JSONResponse
    
    app = FastAPI(title="StyleTTS2 Test API")
    
    @app.get("/")
    async def root():
        return {"message": "Test API is running", "status": "healthy"}
    
    @app.post("/test_upload")
    async def test_upload(
        audio_file: UploadFile = File(...),
        text: str = Form(...)
    ):
        return {
            "status": "success",
            "filename": audio_file.filename,
            "text": text,
            "file_size": len(await audio_file.read())
        }
    
    return app

if __name__ == "__main__":
    print("Testing StyleTTS2 FastAPI setup...")
    print("=" * 50)
    
    # Test basic imports
    print("\n1. Testing basic imports:")
    basic_ok = test_basic_imports()
    
    print("\n2. Testing FastAPI imports:")
    fastapi_ok = test_fastapi_imports()
    
    print("\n3. Testing local imports:")
    local_ok = test_local_imports()
    
    if basic_ok and fastapi_ok:
        print("\n4. Creating test API:")
        try:
            app = create_simple_api()
            print("✓ Test API created successfully")
            
            print("\nStarting test server...")
            print("You can test with:")
            print("curl http://localhost:8000/")
            
            import uvicorn
            uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
            
        except Exception as e:
            print(f"✗ Error creating API: {e}")
    else:
        print("\n✗ Cannot proceed due to import errors")
        sys.exit(1)