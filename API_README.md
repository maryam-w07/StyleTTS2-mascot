# StyleTTS2 FastAPI Wrapper

This repository contains a FastAPI wrapper for the StyleTTS2 inference script that provides viseme generation from audio and text inputs.

## Features

- **RESTful API** for StyleTTS2 viseme generation
- **Audio file support** for .wav and .mp3 formats  
- **Text input** processing with phonemization
- **JSON output** with viseme timing data
- **Error handling** and input validation
- **Simplified inference mode** for testing without full model dependencies

## API Endpoints

### Health Check Endpoints

#### `GET /`
Basic health check endpoint.

**Response:**
```json
{
  "message": "StyleTTS2 Viseme API is running",
  "status": "healthy"
}
```

#### `GET /health`
Detailed health check with model status.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "full_inference": false,
  "simplified_mode": true
}
```

### Viseme Generation Endpoints

#### `POST /generate_visemes`
Generate viseme data with full response metadata.

**Parameters:**
- `audio_file` (file): Audio file in .wav or .mp3 format
- `text` (form data): Text content corresponding to the audio

**Response:**
```json
{
  "status": "success",
  "filename": "test_audio.wav",
  "text": "Hello world",
  "inference_mode": "simplified",
  "visemes": [
    {"offset": 0.0, "visemeId": 12},
    {"offset": 0.5, "visemeId": 7}
  ]
}
```

#### `POST /generate_visemes_json`
Generate viseme data with array-only response (compatible with original script).

**Parameters:**
- `audio_file` (file): Audio file in .wav or .mp3 format
- `text` (form data): Text content corresponding to the audio

**Response:**
```json
[
  {"offset": 0.0, "visemeId": 12},
  {"offset": 0.5, "visemeId": 7}
]
```

## Viseme IDs

The API returns viseme IDs based on phoneme-to-viseme mapping:

| Viseme ID | Description | Example Phonemes |
|-----------|-------------|------------------|
| 0 | Silence/Rest | silence, punctuation |
| 1 | a, æ, ʌ, ə, ɚ | cat, but, about |
| 2 | ɑ, a, aː, ai, au | car, father |
| 3 | ɔ, ɔy, ɔɪ, ɔ̃ | dog, boy |
| 4 | eɪ, ɛ, e, ɐ, œ | bed, day |
| 5 | ɜː, ɝ | bird, burn |
| 6 | i, ɪ, iː, j, ju, y | see, bit, yes |
| 7 | oʊ, u, uː, ʊ, w | go, put, we |
| 8 | o, oː | boat |
| 9 | aʊ, aʊɹ | how, hour |
| 10 | ɔɪ | boy |
| 11 | aɪ, aɪɹ | my, fire |
| 12 | h, x, ç | hat |
| 13 | ɹ, r, ʀ, ʁ | red |
| 14 | l, lʲ, ʎ | love |
| 15 | s, z, ʂ, ʐ | see, zoo |
| 16 | ʃ, ʒ, ʤ | she, measure |
| 17 | ð | this |
| 18 | f, v | fox, voice |
| 19 | θ, t, d, n, ʔ | think, top, dog, no |
| 20 | k, g, ŋ | cat, go, sing |
| 21 | p, b, m | pat, bat, map |

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the API server:
```bash
python api_simple.py
```

The server will start on `http://localhost:8000`

## Usage Examples

### Using curl

```bash
# Test the health endpoint
curl -X GET http://localhost:8000/health

# Generate visemes
curl -X POST "http://localhost:8000/generate_visemes" \
  -H "Content-Type: multipart/form-data" \
  -F "audio_file=@your_audio.wav" \
  -F "text=Hello world this is a test"

# Get just the viseme array
curl -X POST "http://localhost:8000/generate_visemes_json" \
  -H "Content-Type: multipart/form-data" \
  -F "audio_file=@your_audio.wav" \
  -F "text=Hello world"
```

### Using Python requests

```python
import requests

# Generate visemes
with open('your_audio.wav', 'rb') as audio_file:
    response = requests.post(
        'http://localhost:8000/generate_visemes',
        files={'audio_file': audio_file},
        data={'text': 'Hello world this is a test'}
    )
    
result = response.json()
print(result['visemes'])
```

### Using JavaScript/fetch

```javascript
const formData = new FormData();
formData.append('audio_file', audioFile); // File object
formData.append('text', 'Hello world this is a test');

fetch('http://localhost:8000/generate_visemes', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('Visemes:', data.visemes);
});
```

## API Testing

You can use the included test script to verify the API is working:

```bash
python test_api.py
```

## File Structure

- `api_simple.py` - Main FastAPI application with simplified inference
- `api.py` - Full FastAPI application (requires complete model setup)
- `inference.py` - Modified original inference script with API-friendly functions
- `test_api.py` - Test script for API functionality
- `requirements.txt` - Python dependencies
- `phoneme_viseme.json` - Phoneme to viseme mapping configuration

## Inference Modes

The API supports two inference modes:

1. **Simplified Mode**: Uses basic phoneme-to-viseme mapping without full StyleTTS2 models. Good for testing and development.

2. **Full Mode**: Uses complete StyleTTS2 pipeline with forced alignment. Requires trained models and full dependency installation.

## Error Handling

The API includes comprehensive error handling:

- **400 Bad Request**: Invalid file format, missing text, etc.
- **500 Internal Server Error**: Processing errors, model issues
- **503 Service Unavailable**: Models not loaded/available

## Development

To extend the API:

1. Modify `api_simple.py` for new endpoints
2. Update inference functions in `inference.py`
3. Add new viseme mappings in `phoneme_viseme.json`
4. Update tests in `test_api.py`

## License

[Include your license information here]