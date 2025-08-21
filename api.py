#!/usr/bin/env python3
"""
FastAPI wrapper for StyleTTS2 inference script.
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

# Import from the original inference script
from inference import initialize_models, inference_viseme_json_standalone

# Initialize FastAPI app
app = FastAPI(
    title="StyleTTS2 Viseme API",
    description="API for generating viseme data from audio and text using StyleTTS2",
    version="1.0.0"
)

# Global model storage
models = None

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global models
    try:
        print("Initializing StyleTTS2 models...")
        models = initialize_models()
        print("Models initialized successfully!")
    except Exception as e:
        print(f"Error initializing models: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "StyleTTS2 Viseme API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    global models
    if models is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": "models not initialized"}
        )
    return {"status": "healthy", "models_loaded": True}

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
        raise HTTPException(status_code=503, detail="Models not initialized")
    
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
        
        # Convert to .wav if necessary (the inference expects .wav)
        if file_extension == '.mp3':
            wav_path = os.path.join(temp_dir, "input.wav")
            # Load mp3 and save as wav
            audio_data, sample_rate = librosa.load(temp_audio_path, sr=None)
            sf.write(wav_path, audio_data, sample_rate)
            temp_audio_path = wav_path
        
        # Generate visemes
        print(f"Processing audio: {audio_file.filename}")
        print(f"Text: {text}")
        
        viseme_data = inference_viseme_json_standalone(
            audio_path=temp_audio_path,
            text=text.strip(),
            models_dict=models
        )
        
        return JSONResponse(
            content={
                "status": "success",
                "filename": audio_file.filename,
                "text": text.strip(),
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
        raise HTTPException(status_code=503, detail="Models not initialized")
    
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
        viseme_data = inference_viseme_json_standalone(
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
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info"
    )