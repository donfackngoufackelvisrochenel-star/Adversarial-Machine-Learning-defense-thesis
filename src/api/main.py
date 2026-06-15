"""
FastAPI application for AML Defense.

Provides REST + WebSocket endpoints for:
  - Listing available models
  - Running predictions (REST + WebSocket streaming)
  - Real-time live inference from test set
  - Simulating FGSM / PGD attacks
  - Applying feature squeezing
  - Uploading new datasets

Run with:  uvicorn src.api.main:app --reload
"""

import asyncio
import json
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, Depends, Header
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from typing import Optional
from pathlib import Path

from configs.config import MODELS_DIR, AUTH_USERNAME, AUTH_PASSWORD, API_KEY
from src.preprocessing.processor import DataProcessor
from src.preprocessing.loader import load_data, clean_data

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

_security = HTTPBasic()


def verify_auth(credentials: HTTPBasicCredentials = Depends(_security)):
    """Reject requests with invalid username/password."""
    if credentials.username != AUTH_USERNAME or credentials.password != AUTH_PASSWORD:
        raise HTTPException(401, "Invalid credentials")
    return credentials.username


def verify_api_key(x_api_key: str = Header(None)):
    """Alternative auth via X-API-Key header for programmatic clients."""
    if x_api_key != API_KEY:
        raise HTTPException(401, "Invalid API key")
    return x_api_key

# ---------------------------------------------------------------------------
# FastAPI app initialisation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AML Defense API",
    version="1.0.0",
    description="REST + WebSocket API for adversarial ML defense — prediction, attack simulation, live inference, and defense evaluation.",
)

# Global caches for loaded models, the data processor, and test data
_models = {}
_processor = None
_X_test = None
_y_test = None


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    """A single sample's features as a dict of column_name → value."""
    features: dict


class PredictionResponse(BaseModel):
    """Prediction result returned by the API."""
    predictions: list
    probabilities: Optional[list] = None
    model_used: str


# ---------------------------------------------------------------------------
# Startup — load saved models and processor from disk
# ---------------------------------------------------------------------------

@app.on_event("startup")
def startup():
    """Load models, processor, and test data into memory."""
    global _processor, _X_test, _y_test

    model_files = list(MODELS_DIR.glob("*.pkl"))
    for mf in model_files:
        stem = mf.stem
        if stem in ("X_train", "X_val", "X_test", "y_train", "y_val", "y_test", "df_preview", "processor"):
            continue
        try:
            _models[stem] = joblib.load(mf)
        except Exception as e:
            print(f"[api] Failed to load {mf.name}: {e}")

    proc_path = MODELS_DIR / "processor.pkl"
    if proc_path.exists():
        _processor = DataProcessor()
        _processor.load(proc_path)

    # Load test set for live inference
    xt_path = MODELS_DIR / "X_test.pkl"
    yt_path = MODELS_DIR / "y_test.pkl"
    if xt_path.exists() and yt_path.exists():
        _X_test = joblib.load(xt_path)
        _y_test = joblib.load(yt_path)
        print(f"[api] Loaded test set: {_X_test.shape}")
    else:
        print("[api] No test set found — live endpoint will be unavailable")

    print(f"[api] Loaded {len(_models)} models: {list(_models.keys())}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
def root():
    """Root endpoint returning API metadata and available endpoints."""
    return {
        "service": "AML Defense API",
        "models_available": list(_models.keys()),
        "test_samples": _X_test.shape[0] if _X_test is not None else 0,
        "authentication": "HTTP Basic Auth (username/password) or X-API-Key header",
        "endpoints_secured": "All endpoints except GET / and GET /live/status require auth",
        "endpoints": {
            "GET /": "this info",
            "GET /models": "list loaded models",
            "POST /predict/{model_name}": "run prediction",
            "GET /live/status": "live inference system status",
            "GET /live/next/{model_name}": "next test sample prediction (polling)",
            "WS /ws/live/{model_name}": "WebSocket real-time streaming",
            "POST /attack/fgsm": "generate FGSM adversarial examples",
            "POST /attack/pgd": "generate PGD adversarial examples",
            "POST /defense/squeeze": "apply feature squeezing",
        },
    }


@app.get("/models")
def list_models(username: str = Depends(verify_auth)):
    """Return the list of model names currently loaded in memory."""
    return {"models": list(_models.keys())}


@app.post("/predict/{model_name}")
def predict(model_name: str, req: PredictionRequest, username: str = Depends(verify_auth)):
    """
    Run a single prediction using the specified model.

    Args:
        model_name: Name of the model (without .pkl extension).
        req: JSON body with 'features' dict.

    Returns:
        PredictionResponse with predicted class and probabilities.
    """
    if model_name not in _models:
        raise HTTPException(404, f"Model '{model_name}' not loaded")

    model = _models[model_name]

    # Convert the input dict to a single-row DataFrame
    df = pd.DataFrame([req.features])

    # Apply the same scaling that was applied during training
    if _processor:
        df, _ = _processor.transform(df)

    preds = model.predict(df).tolist()
    probs = model.predict_proba(df).tolist() if hasattr(model, "predict_proba") else None

    return PredictionResponse(predictions=preds, probabilities=probs, model_used=model_name)


@app.post("/attack/fgsm")
def attack_fgsm(model_name: str, epsilon: float = 0.1, username: str = Depends(verify_auth)):
    """
    Simulate an FGSM attack on the specified model.

    (Currently returns a confirmation — full implementation requires
     sample data to perturb.)
    """
    if model_name not in _models:
        raise HTTPException(404, f"Model '{model_name}' not loaded")
    return {"message": "FGSM attack simulated", "model": model_name, "epsilon": epsilon}


@app.post("/attack/pgd")
def attack_pgd(model_name: str, epsilon: float = 0.1, alpha: float = 0.01, num_iter: int = 10,
               username: str = Depends(verify_auth)):
    """
    Simulate a PGD attack on the specified model.

    (Currently returns a confirmation — full implementation requires
     sample data to perturb.)
    """
    if model_name not in _models:
        raise HTTPException(404, f"Model '{model_name}' not loaded")
    return {"message": "PGD attack simulated", "model": model_name, "epsilon": epsilon, "alpha": alpha, "iter": num_iter}


@app.post("/defense/squeeze")
def squeeze_defense(squeeze_type: str = "bit_depth", bit_depth: int = 4,
                    username: str = Depends(verify_auth)):
    """
    Apply feature squeezing as a defense.

    (Currently returns a confirmation — full implementation would
     require features to squeeze.)
    """
    return {"message": f"Feature squeezing ({squeeze_type}) applied", "bit_depth": bit_depth}


# ---------------------------------------------------------------------------
# Real-time / Live Inference Endpoints
# ---------------------------------------------------------------------------


@app.get("/live/status")
def live_status():
    """Return the status of the live inference system."""
    return {
        "models_available": list(_models.keys()),
        "test_samples": _X_test.shape[0] if _X_test is not None else 0,
        "streamer_ready": _X_test is not None and len(_models) > 0,
    }


@app.get("/live/next/{model_name}")
def live_next(model_name: str, attack: bool = False, epsilon: float = 0.1,
              attack_type: str = "fgsm",
              username: str = Depends(verify_auth)):
    """
    Fetch the next test sample, run prediction, and return the result.

    This is the polling-based real-time endpoint (compatible with Streamlit).
    A new sample is returned on each call, cycling through the test set.

    Query params:
        attack      : whether to apply adversarial perturbation
        epsilon     : attack strength (default 0.1)
        attack_type : "fgsm" (default) or "pgd"
    """
    global _live_idx
    if not hasattr(live_next, "_idx"):
        live_next._idx = 0

    if _X_test is None or _y_test is None:
        raise HTTPException(503, "Test set not loaded. Run the pipeline first.")
    if model_name not in _models:
        raise HTTPException(404, f"Model '{model_name}' not loaded. Available: {list(_models.keys())}")

    model = _models[model_name]
    idx = live_next._idx % _X_test.shape[0]
    live_next._idx += 1

    sample = _X_test.iloc[idx]
    true_label = int(_y_test.iloc[idx])

    # Apply adversarial perturbation if requested
    if attack and epsilon > 0:
        df = pd.DataFrame([sample])
        if attack_type == "pgd":
            from src.attacks.pgd import pgd_attack
            df_adv = pgd_attack(model, df, pd.Series([true_label]), epsilon=epsilon)
        else:
            from src.attacks.fgsm import fgsm_attack
            df_adv = fgsm_attack(model, df, pd.Series([true_label]), epsilon=epsilon)
        sample = df_adv.iloc[0]

    # Predict
    df_in = pd.DataFrame([sample])
    pred = int(model.predict(df_in)[0])
    prob = float(np.max(model.predict_proba(df_in)[0]))
    correct = pred == true_label

    return {
        "sample_index": int(idx),
        "true_label": true_label,
        "prediction": pred,
        "confidence": round(prob, 4),
        "correct": correct,
        "attack_active": attack,
        "attack_type": attack_type if attack else "none",
        "epsilon": epsilon if attack else 0.0,
    }


# ---------------------------------------------------------------------------
# WebSocket — real-time streaming inference
# ---------------------------------------------------------------------------
# Connect with:  ws://localhost:8000/ws/live/{model_name}
# The server streams one prediction per row from the test set.
# The client can send JSON messages to control the stream:
#   {"cmd": "stop"}      — pause streaming
#   {"cmd": "resume"}    — resume streaming
#   {"cmd": "attack", "epsilon": 0.1}  — enable attack
#   {"cmd": "noattack"}  — disable attack
#   {"cmd": "reset"}     — reset to sample 0


@app.websocket("/ws/live/{model_name}")
async def websocket_live(websocket: WebSocket, model_name: str):
    # WebSocket auth: client must send credentials as first message
    await websocket.accept()
    try:
        auth_msg = await asyncio.wait_for(websocket.receive_json(), timeout=5.0)
        if auth_msg.get("username") != AUTH_USERNAME or auth_msg.get("password") != AUTH_PASSWORD:
            await websocket.send_json({"error": "Authentication failed"})
            await websocket.close(code=4001)
            return
    except (asyncio.TimeoutError, ValueError):
        await websocket.send_json({"error": "Authentication required — send JSON {username, password} as first message"})
        await websocket.close(code=4001)
        return

    if model_name not in _models:
        await websocket.send_json({"error": f"Model '{model_name}' not found"})
        await websocket.close()
        return

    if _X_test is None:
        await websocket.send_json({"error": "Test set not loaded"})
        await websocket.close()
        return

    model = _models[model_name]
    idx = 0
    n = _X_test.shape[0]
    paused = False
    attack_active = False
    attack_type = "fgsm"
    epsilon = 0.1

    await websocket.send_json({
        "status": "connected",
        "model": model_name,
        "total_samples": n,
        "message": "Streaming live predictions. Send JSON commands to control.",
    })

    try:
        while True:
            # Check for incoming control messages (non-blocking)
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                msg = json.loads(data)
                cmd = msg.get("cmd", "")
                if cmd == "stop":
                    paused = True
                    await websocket.send_json({"status": "paused"})
                elif cmd == "resume":
                    paused = False
                    await websocket.send_json({"status": "resumed"})
                elif cmd == "attack":
                    attack_active = True
                    epsilon = float(msg.get("epsilon", 0.1))
                    attack_type = msg.get("attack_type", "fgsm")
                    await websocket.send_json({"status": "attack_on", "attack_type": attack_type, "epsilon": epsilon})
                elif cmd == "noattack":
                    attack_active = False
                    await websocket.send_json({"status": "attack_off"})
                elif cmd == "reset":
                    idx = 0
                    await websocket.send_json({"status": "reset"})
            except asyncio.TimeoutError:
                pass

            if paused:
                await asyncio.sleep(0.1)
                continue

            # Get next sample (loop if needed)
            idx = idx % n
            sample = _X_test.iloc[idx]
            true_label = int(_y_test.iloc[idx])

            # Attack
            df = pd.DataFrame([sample])
            if attack_active and epsilon > 0:
                atype = attack_type if attack_active else "fgsm"
                if atype == "pgd":
                    from src.attacks.pgd import pgd_attack
                    df_adv = pgd_attack(model, df, pd.Series([true_label]), epsilon=epsilon)
                else:
                    from src.attacks.fgsm import fgsm_attack
                    df_adv = fgsm_attack(model, df, pd.Series([true_label]), epsilon=epsilon)
                df = df_adv

            # Predict
            pred = int(model.predict(df)[0])
            prob = float(np.max(model.predict_proba(df)[0]))
            correct = pred == true_label

            await websocket.send_json({
                "sample_index": int(idx),
                "true_label": true_label,
                "prediction": pred,
                "confidence": round(prob, 4),
                "correct": correct,
                "attack_active": attack_active,
                "attack_type": attack_type if attack_active else "none",
                "epsilon": epsilon if attack_active else 0.0,
            })

            idx += 1
            await asyncio.sleep(0.05)  # ~20 samples/sec

    except WebSocketDisconnect:
        print(f"[api] WebSocket disconnected for {model_name}")


# ---------------------------------------------------------------------------
# File Upload
# ---------------------------------------------------------------------------


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...), username: str = Depends(verify_auth)):
    """
    Upload a dataset file to data/raw/.

    Accepts CSV, TXT, ZIP, or GZ files.
    """
    dest = Path("data/raw") / file.filename
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)
    return {"message": f"Uploaded {file.filename}", "size": len(content)}
