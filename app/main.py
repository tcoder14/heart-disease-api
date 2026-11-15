import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import HeartInput, InfoOut, PredictionOut

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        Path(__file__).resolve().parent.parent / "model" / "heart_model.joblib",
    )
)

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup, not on every request.
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at {MODEL_PATH}. Run `python train.py` first.")
    state["bundle"] = joblib.load(MODEL_PATH)
    yield
    state.clear()


app = FastAPI(
    title="Heart Disease Prediction API",
    description="Serves predictions from a classifier trained on the Heart Disease dataset.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"message": "Heart Disease Prediction API. See /docs for Swagger UI."}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "bundle" in state}


@app.get("/info", response_model=InfoOut)
def info():
    bundle = state["bundle"]
    return InfoOut(
        model_type=type(bundle["model"]).__name__,
        n_features=len(bundle["features"]),
        features=bundle["features"],
        test_accuracy=bundle.get("test_accuracy"),
    )


@app.post("/predict", response_model=PredictionOut)
def predict(payload: HeartInput):
    bundle = state["bundle"]
    try:
        # Build a DataFrame with columns in the exact order used for training.
        X = pd.DataFrame([payload.model_dump()])[bundle["features"]]
        model = bundle["model"]
        proba = float(model.predict_proba(X)[0][1])
        pred = bool(model.predict(X)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
    return PredictionOut(heart_disease=pred, probability=round(proba, 4))
