import whisper
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
from enum import Enum
from typing import Optional
app = FastAPI()

# Model state
class ModelState(str, Enum):
    LOADED = "loaded"
    UNLOADED = "unloaded"

model = None
model_state = ModelState.UNLOADED
model_name = "base"

def load_model():
    global model, model_state
    if model_state == ModelState.UNLOADED:
        model = whisper.load_model(model_name)
        model_state = ModelState.LOADED
        return True
    return False

def unload_model():
    global model, model_state
    if model_state == ModelState.LOADED:
        model = None
        model_state = ModelState.UNLOADED
        return True
    return False

@app.post("/asr")
async def transcribe_audio(file: UploadFile = File(...)):
    if model_state == ModelState.UNLOADED:
        return JSONResponse(
            status_code=503,
            content={"error": "Model is not loaded. Use /toggle to load the model."}
        )
    
    try:
        # Save uploaded file temporarily
        temp_file = f"/tmp/{file.filename}"
        with open(temp_file, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Transcribe
        result = model.transcribe(temp_file, initial_prompt="以下是繁體中文或是英文的的逐字稿。The following is a verbatim transcript in Traditional Chinese or English.")
        
        # Clean up
        os.remove(temp_file)
        
        return JSONResponse(content={
            "text": result["text"],
            "language": result.get("language", "unknown")
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/status")
async def get_status():
    return JSONResponse(content={
        "model_state": model_state,
        "model_name": model_name
    })

@app.post("/toggle")
async def toggle_model():
    if model_state == ModelState.UNLOADED:
        load_model()
        return JSONResponse(content={
            "message": "Model loaded successfully",
            "model_state": model_state
        })
    else:
        unload_model()
        return JSONResponse(
            status_code=200,
            content={
            "message": "Model unloaded successfully",
            "model_state": model_state,
            "success": True
            }
        )
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)