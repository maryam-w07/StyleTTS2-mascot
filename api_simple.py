#!/usr/bin/env python3
"""
Simplified FastAPI wrapper for StyleTTS2 inference script.
Provides an endpoint that accepts audio files (.wav/.mp3) and text, 
returns viseme JSON output.
"""

import os
import tempfile
import shutil
from typing import Optional
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import librosa
import soundfile as sf
import json

# Initialize FastAPI app
app = FastAPI(
    title="StyleTTS2 Viseme API",
    description="API for generating viseme data from audio and text using StyleTTS2",
    version="1.0.0"
)

# Global model storage
models = None
models_loading = False

def simple_viseme_inference(audio_path, text):
    """
    Simplified viseme inference that returns mock data for testing
    until the full model pipeline is working
    """
    # For now, return a simplified mock response based on text length
    # This allows us to test the API structure
    
    words = text.split()
    visemes = []
    time_offset = 0.0
    
    # Simple mapping based on first letter of each word
    letter_to_viseme = {
        'a': 1, 'e': 4, 'i': 6, 'o': 8, 'u': 7,
        'b': 21, 'p': 21, 'm': 21,
        'f': 18, 'v': 18,
        't': 19, 'd': 19, 'n': 19, 'l': 14,
        's': 15, 'z': 15,
        'r': 13, 'w': 7, 'y': 6, 'h': 12
    }
    
    for word in words:
        if word:
            # Get viseme for first letter
            first_letter = word[0].lower()
            viseme_id = letter_to_viseme.get(first_letter, 0)
            
            visemes.append({
                "offset": round(time_offset, 3),
                "visemeId": viseme_id
            })
            
            # Assume each word takes about 500ms
            time_offset += 0.5
    
    return visemes

async def try_load_models():
    """Attempt to load the full StyleTTS2 models"""
    global models, models_loading
    
    if models_loading:
        return False
        
    try:
        models_loading = True
        print("Attempting to load StyleTTS2 models...")
        
        # Try to import and initialize the full inference pipeline
        from inference import initialize_models, inference_viseme_json_standalone
        
        models = initialize_models()
        models['inference_function'] = inference_viseme_json_standalone
        print("StyleTTS2 models loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Could not load full StyleTTS2 models: {e}")
        print("Using simplified inference mode...")
        models = {'simplified': True}
        return False
    finally:
        models_loading = False

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    await try_load_models()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "StyleTTS2 Viseme API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    global models
    
    status = {
        "status": "healthy",
        "models_loaded": models is not None,
        "full_inference": models is not None and 'inference_function' in models,
        "simplified_mode": models is not None and models.get('simplified', False)
    }
    
    if models is None:
        status["status"] = "models_loading"
        
    return status

@app.post("/generate_visemes")
async def generate_visemes(
    audio_file: UploadFile = File(..., description="Audio file (.wav or .mp3)"),
    text: str = Form(..., description="Text content corresponding to the audio")
):
    """
    Generate viseme data from audio and text.
    
    Args:
        audio_file: Upload audio file in .wav or .mp3 format
        text: Text content that corresponds to the audio
        
    Returns:
        JSON response with viseme timing data
    """
    global models
    
    if models is None:
        # Try to load models
        await try_load_models()
        if models is None:
            raise HTTPException(status_code=503, detail="Models not available")
    
    # Validate file type
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_extension = os.path.splitext(audio_file.filename)[1].lower()
    if file_extension not in ['.wav', '.mp3']:
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file format. Please upload .wav or .mp3 files."
        )
    
    # Validate text input
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty")
    
    # Create temporary file to store uploaded audio
    temp_dir = tempfile.mkdtemp()
    temp_audio_path = None
    
    try:
        # Save uploaded file
        temp_audio_path = os.path.join(temp_dir, f"input{file_extension}")
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        # Convert to .wav if necessary
        if file_extension == '.mp3':
            wav_path = os.path.join(temp_dir, "input.wav")
            # Load mp3 and save as wav
            audio_data, sample_rate = librosa.load(temp_audio_path, sr=None)
            sf.write(wav_path, audio_data, sample_rate)
            temp_audio_path = wav_path
        
        # Generate visemes
        print(f"Processing audio: {audio_file.filename}")
        print(f"Text: {text}")
        
        if models.get('simplified', False):
            # Use simplified inference
            viseme_data = simple_viseme_inference(temp_audio_path, text.strip())
            inference_mode = "simplified"
        else:
            # Use full StyleTTS2 inference
            viseme_data = models['inference_function'](
                audio_path=temp_audio_path,
                text=text.strip(),
                models_dict=models
            )
            inference_mode = "full"
        
        return JSONResponse(
            content={
                "status": "success",
                "filename": audio_file.filename,
                "text": text.strip(),
                "inference_mode": inference_mode,
                "visemes": viseme_data
            }
        )
        
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        # Clean up temporary files
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

@app.post("/generate_visemes_json")
async def generate_visemes_json(
    audio_file: UploadFile = File(..., description="Audio file (.wav or .mp3)"),
    text: str = Form(..., description="Text content corresponding to the audio")
):
    """
    Alternative endpoint that returns just the viseme array (compatible with original script output)
    """
    global models
    
    if models is None:
        await try_load_models()
        if models is None:
            raise HTTPException(status_code=503, detail="Models not available")
    
    # Validate inputs (same as above)
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_extension = os.path.splitext(audio_file.filename)[1].lower()
    if file_extension not in ['.wav', '.mp3']:
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file format. Please upload .wav or .mp3 files."
        )
    
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty")
    
    # Create temporary file
    temp_dir = tempfile.mkdtemp()
    temp_audio_path = None
    
    try:
        # Save and process file (same as above)
        temp_audio_path = os.path.join(temp_dir, f"input{file_extension}")
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        if file_extension == '.mp3':
            wav_path = os.path.join(temp_dir, "input.wav")
            audio_data, sample_rate = librosa.load(temp_audio_path, sr=None)
            sf.write(wav_path, audio_data, sample_rate)
            temp_audio_path = wav_path
        
        # Generate visemes
        if models.get('simplified', False):
            viseme_data = simple_viseme_inference(temp_audio_path, text.strip())
        else:
            viseme_data = models['inference_function'](
                audio_path=temp_audio_path,
                text=text.strip(),
                models_dict=models
            )
        
        # Return just the viseme array (as per original script)
        return viseme_data
        
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        # Clean up
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    # Run the FastAPI server
    print("Starting StyleTTS2 Viseme API...")
    uvicorn.run(
        "api_simple:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info"
    )