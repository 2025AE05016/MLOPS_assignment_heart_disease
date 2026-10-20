from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
import time
import uuid

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from prometheus_client import Counter, Histogram, CONTENT_TYPE_LATEST, generate_latest
from .data import FEATURES

logger = logging.getLogger("heartops")
logging.basicConfig(level=logging.INFO)
REQUESTS = Counter("heartops_requests_total", "API responses", ["route", "status"])
LATENCY = Histogram("heartops_request_seconds", "API latency", ["route"])
PREDICTIONS = Counter("heartops_predictions_total", "Predicted classes", ["prediction"])
PROBABILITY = Histogram("heartops_disease_probability", "Predicted disease probability",
                        buckets=[i / 10 for i in range(1, 11)])


class Patient(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    age: float = Field(ge=1, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=1, le=4)
    trestbps: float = Field(gt=0, le=300)
    chol: float = Field(gt=0, le=700)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(gt=0, le=250)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=0, le=10)
    slope: int = Field(ge=1, le=3)
    ca: int | None = Field(default=None, ge=0, le=3)
    thal: int | None = None


@asynccontextmanager
async def lifespan(app):
    app.state.model = joblib.load(Path(os.getenv("MODEL_PATH", "artifacts/model.joblib")))
    yield


app = FastAPI(title="HeartOps educational classifier", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    # Avoid echoing patient data or nonfinite numbers into JSON error responses.
    errors = [{"loc": error["loc"], "msg": error["msg"], "type": error["type"]}
              for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.middleware("http")
async def observe(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        route = route.path if route else "unmatched"
        elapsed = time.perf_counter() - start
        REQUESTS.labels(route, str(status)).inc()
        LATENCY.labels(route).observe(elapsed)
        logger.info(json.dumps({"request_id": str(uuid.uuid4()), "route": route,
                                "status": status, "seconds": elapsed}))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if not hasattr(app.state, "model"):
        raise HTTPException(503, "Model unavailable")
    return {"model_loaded": True}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), headers={"Content-Type": CONTENT_TYPE_LATEST})


@app.post("/predict")
def predict(patient: Patient):
    if patient.thal not in (None, 3, 6, 7):
        raise HTTPException(422, "thal must be 3, 6, 7 or null (original UCI codes)")
    frame = pd.DataFrame([patient.model_dump()], columns=FEATURES, dtype=float)
    probability = float(app.state.model.predict_proba(frame)[0, 1])
    label = int(probability >= 0.5)
    PREDICTIONS.labels(str(label)).inc()
    PROBABILITY.observe(probability)
    return {"prediction": label, "disease_probability": probability,
            "confidence": probability if label else 1 - probability,
            "threshold": 0.5}
