import sys, os
import tempfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from inference_module import load_model, inference_viseme_json

# Ensure parent folder is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize FastAPI app
app = FastAPI(title="Viseme Inference API")

# ---------- Load model on startup ----------
print("[INFO] Loading model...")
model_bundle = load_model(
    config_path=os.path.join(os.path.dirname(__file__), "..", "config_ft.yml"),
    device="cuda"
)
print("[INFO] Model ready.")

# ---------- Root endpoint ----------
@app.get("/")
async def root():
    return {"message": "Viseme Inference API is running!"}

# ---------- Inference endpoint ----------
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
        return JSONResponse(status_code=500, content={"error": str(e)})

    return {"visemes": viseme_json}

# ---------- Run app ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
