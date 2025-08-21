from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from .inference_module import load_model, inference_viseme_json
import tempfile
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="Viseme Inference API")

# Load model once on startup
print("[INFO] Loading model...")
model_bundle = load_model(config_path="Configs/config_ft.yml", device="cuda")
print("[INFO] Model ready.")

@app.post("/infer/")
async def infer_visemes(
    file: UploadFile = File(..., description="Audio file (.wav or .mp3)"),
    text: str = Form(..., description="Input text"),
):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name


    # Run inference
    try:
        viseme_json = inference_viseme_json(
            model_bundle=model_bundle,
            audio_path=tmp_path,
            text=text,
        )
    except Exception as e:
        return {"error": str(e)}

    return {"visemes": viseme_json}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
