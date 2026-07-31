from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import argparse
import sys

app = FastAPI(title="Render Cloud MLOps API", version="1.0.0")

# ==============================================================================
# Model Schemas
# ==============================================================================
class PredictRequest(BaseModel):
    feature_1: float
    feature_2: float
    feature_3: str = "default_metric"

class PredictResponse(BaseModel):
    prediction_score: float
    status: str

# ==============================================================================
# ML Logic / Inference Handler
# ==============================================================================
def mock_ml_inference(f1: float, f2: float) -> float:
    # A dummy logic representing a model serving layer
    return round((f1 * 0.72) + (f2 * 0.28), 4)

# ==============================================================================
# FastAPI Routes
# ==============================================================================
@app.get("/")
def read_root():
    return {"message": "Welcome to the Render Deployment MLOps API! System is functional."}

@app.get("/health")
def health_check():
    # Crucial for Render Blueprints Health Probe Checks
    return {"status": "ok", "service": "online"}

@app.post("/predict", response_model=PredictResponse)
def run_prediction(req: PredictRequest):
    if req.feature_1 < 0 or req.feature_2 < 0:
        raise HTTPException(status_code=400, detail="Input features must be dynamically positive.")
    
    score = mock_ml_inference(req.feature_1, req.feature_2)
    return PredictResponse(prediction_score=score, status="success")

# ==============================================================================
# CLI Launcher
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['serve', 'test'], default='serve')
    args, _ = parser.parse_known_args()
    
    if args.mode == 'serve':
        print("Starting FastAPI Engine on Port 8000...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    elif args.mode == 'test':
        import pytest
        sys.exit(pytest.main(["-v"]))
