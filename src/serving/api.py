"""
api.py - FastAPI Model Serving with A/B Testing, Rate Limiting & Monitoring

Covers:
- RESTful API with FastAPI
- Model loading from MLflow registry
- A/B testing / canary deployments
- Request/response validation (Pydantic)
- Prometheus metrics integration
- Health checks and readiness probes
- Batch inference endpoint
- Async processing
"""

import os
import time
import asyncio
import logging
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime

import torch
import numpy as np
import mlflow
import mlflow.pytorch
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Prometheus Metrics ─────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    'prediction_requests_total',
    'Total number of prediction requests',
    ['model_version', 'sentiment', 'status']
)

REQUEST_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Prediction request latency in seconds',
    ['model_version'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

MODEL_CONFIDENCE = Histogram(
    'prediction_confidence',
    'Model prediction confidence scores',
    ['model_version', 'sentiment'],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
)

ACTIVE_REQUESTS = Gauge(
    'active_prediction_requests',
    'Number of currently active prediction requests'
)

BATCH_SIZE = Histogram(
    'batch_prediction_size',
    'Size of batch prediction requests',
    buckets=[1, 5, 10, 25, 50, 100]
)


# ─── Pydantic Schemas ────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    text: str
    model_version: str = "production"  # production, staging, champion, challenger
    request_id: Optional[str] = None

    @validator('text')
    def text_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        if len(v) > 10000:
            raise ValueError('Text too long (max 10000 characters)')
        return v.strip()


class BatchPredictionRequest(BaseModel):
    texts: List[str]
    model_version: str = "production"

    @validator('texts')
    def validate_batch(cls, v):
        if len(v) == 0:
            raise ValueError('Batch cannot be empty')
        if len(v) > 100:
            raise ValueError('Batch size too large (max 100)')
        return v


class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]
    model_version: str
    inference_time_ms: float
    request_id: Optional[str]
    timestamp: str


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_time_ms: float
    batch_size: int


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    device: str
    timestamp: str


# ─── Model Registry / Loader ─────────────────────────────────────────────────

class ModelRegistry:
    """
    Manages multiple model versions for A/B testing and canary deployments.
    Loads models from MLflow registry or local checkpoints.
    """

    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.label_maps = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.routing_config = {
            "production": 0.9,
            "staging": 0.1
        }
        self.request_counts = {}
        logger.info(f"ModelRegistry initialized on {self.device}")

    def load_model_from_mlflow(self, model_name: str, stage: str = "Production") -> bool:
        """Load model from MLflow registry."""
        try:
            model_uri = f"models:/{model_name}/{stage}"
            logger.info(f"Loading model from MLflow: {model_uri}")
            model = mlflow.pytorch.load_model(model_uri, map_location=self.device)
            model.eval()
            self.models[stage.lower()] = model
            logger.info(f"✅ Model loaded: {model_name} ({stage})")
            return True
        except Exception as e:
            logger.warning(f"Could not load model from MLflow: {e}")
            return False

    def load_local_model(self, model, version_name: str, tokenizer=None, label_map=None):
        """Register a locally created model."""
        model.eval()
        model.to(self.device)
        self.models[version_name] = model
        if tokenizer:
            self.tokenizers[version_name] = tokenizer
        if label_map:
            self.label_maps[version_name] = label_map
        self.request_counts[version_name] = 0
        logger.info(f"✅ Local model registered: {version_name}")

    def get_model(self, version: str = "production"):
        """Get model for a given version, with fallback."""
        if version in self.models:
            return self.models[version]
        # Fallback
        if self.models:
            fallback = list(self.models.keys())[0]
            logger.warning(f"Model '{version}' not found, using '{fallback}'")
            return self.models[fallback]
        raise ValueError("No models loaded in registry")

    def route_request(self, preferred_version: str = "auto") -> str:
        """Route request to appropriate model version (A/B testing)."""
        if preferred_version != "auto" and preferred_version in self.models:
            return preferred_version

        # Weighted routing for A/B testing
        versions = [v for v in self.routing_config if v in self.models]
        if not versions:
            return list(self.models.keys())[0]

        weights = [self.routing_config[v] for v in versions]
        total = sum(weights)
        weights = [w / total for w in weights]

        chosen = np.random.choice(versions, p=weights)
        self.request_counts[chosen] = self.request_counts.get(chosen, 0) + 1
        return chosen

    @property
    def loaded_models(self) -> List[str]:
        return list(self.models.keys())


# ─── Text Preprocessor ───────────────────────────────────────────────────────

class InferencePreprocessor:
    """Lightweight preprocessor for inference."""

    def __init__(self, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def preprocess(self, text: str) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {k: v for k, v in encoding.items()}


# ─── FastAPI App ─────────────────────────────────────────────────────────────

# Global registry
registry = ModelRegistry()

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def predict_single(
    text: str,
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    label_map: Dict[int, str]
) -> Dict[str, Any]:
    """Run inference on a single text."""
    preprocessor = InferencePreprocessor(tokenizer)
    inputs = preprocessor.preprocess(text)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = outputs['probabilities'][0].cpu().numpy()
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    sentiment = label_map[pred_idx]

    probabilities = {
        label_map[i]: float(p)
        for i, p in enumerate(probs)
    }

    return {
        'sentiment': sentiment,
        'confidence': confidence,
        'probabilities': probabilities
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan: startup & shutdown logic."""
    logger.info("🚀 Starting ML API server...")

    # Try to load models from MLflow
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)

    # Load from MLflow registry (if available)
    for model_name in ["bert-sentiment", "cnn-sentiment", "lstm-sentiment"]:
        for stage in ["Production", "Staging"]:
            registry.load_model_from_mlflow(model_name, stage)

    # If no MLflow models, load a demo model
    if not registry.loaded_models:
        logger.warning("No MLflow models found. Loading demo model...")
        try:
            from transformers import AutoTokenizer
            from src.models.bert_model import TransformerClassifier

            demo_model = TransformerClassifier(
                model_name="distilbert-base-uncased",
                num_classes=3
            )
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            registry.load_local_model(
                demo_model, "production",
                tokenizer=tokenizer,
                label_map=LABEL_MAP
            )
        except Exception as e:
            logger.error(f"Could not load demo model: {e}")

    logger.info(f"✅ API ready. Models loaded: {registry.loaded_models}")
    yield
    logger.info("Shutting down API server...")


app = FastAPI(
    title="🤖 NLP Sentiment Classification API",
    description="""
    Production ML API for Sentiment Analysis
    
    ## Features
    - Multiple model versions (BERT, CNN, LSTM, Ensemble)
    - A/B testing with configurable traffic split  
    - Real-time Prometheus metrics
    - Batch inference support
    - Model health monitoring
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ─── Middleware ──────────────────────────────────────────────────────────────

@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track all requests with timing."""
    ACTIVE_REQUESTS.inc()
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    ACTIVE_REQUESTS.dec()
    response.headers["X-Process-Time"] = str(duration)
    return response


# ─── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "🤖 NLP Sentiment API is running!",
        "docs": "/docs",
        "metrics": "/metrics",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint for Kubernetes readiness/liveness probes."""
    if not registry.loaded_models:
        raise HTTPException(status_code=503, detail="No models loaded")

    return HealthResponse(
        status="healthy",
        models_loaded=registry.loaded_models,
        device=str(registry.device),
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/metrics", tags=["System"])
async def get_metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(request: PredictionRequest):
    """
    Single text sentiment prediction.
    
    Supports A/B testing by specifying model_version:
    - production: main model (90% traffic)
    - staging: challenger model (10% traffic)
    - auto: automatic routing based on traffic split
    """
    start_time = time.time()
    ACTIVE_REQUESTS.inc()

    try:
        # Route to appropriate model version
        model_version = registry.route_request(request.model_version)
        model = registry.get_model(model_version)
        tokenizer = registry.tokenizers.get(model_version,
                     registry.tokenizers.get(list(registry.tokenizers.keys())[0])
                     if registry.tokenizers else None)
        label_map = registry.label_maps.get(model_version, LABEL_MAP)

        if tokenizer is None:
            raise HTTPException(status_code=503, detail="Tokenizer not loaded")

        # Run inference
        result = predict_single(
            request.text, model, tokenizer, registry.device, label_map
        )

        inference_time = (time.time() - start_time) * 1000

        # Update Prometheus metrics
        REQUEST_COUNT.labels(
            model_version=model_version,
            sentiment=result['sentiment'],
            status="success"
        ).inc()
        REQUEST_LATENCY.labels(model_version=model_version).observe(
            inference_time / 1000
        )
        MODEL_CONFIDENCE.labels(
            model_version=model_version,
            sentiment=result['sentiment']
        ).observe(result['confidence'])

        return PredictionResponse(
            text=request.text[:200],  # Truncate for response
            sentiment=result['sentiment'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            model_version=model_version,
            inference_time_ms=round(inference_time, 2),
            request_id=request.request_id,
            timestamp=datetime.utcnow().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        REQUEST_COUNT.labels(
            model_version=request.model_version,
            sentiment="unknown",
            status="error"
        ).inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        ACTIVE_REQUESTS.dec()


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction for multiple texts."""
    start_time = time.time()
    BATCH_SIZE.observe(len(request.texts))

    model_version = registry.route_request(request.model_version)
    model = registry.get_model(model_version)
    tokenizer = list(registry.tokenizers.values())[0] if registry.tokenizers else None
    label_map = registry.label_maps.get(model_version, LABEL_MAP)

    if tokenizer is None:
        raise HTTPException(status_code=503, detail="Tokenizer not loaded")

    predictions = []
    for text in request.texts:
        t0 = time.time()
        result = predict_single(text, model, tokenizer, registry.device, label_map)
        inf_time = (time.time() - t0) * 1000

        predictions.append(PredictionResponse(
            text=text[:200],
            sentiment=result['sentiment'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            model_version=model_version,
            inference_time_ms=round(inf_time, 2),
            request_id=None,
            timestamp=datetime.utcnow().isoformat()
        ))

    total_time = (time.time() - start_time) * 1000

    return BatchPredictionResponse(
        predictions=predictions,
        total_time_ms=round(total_time, 2),
        batch_size=len(request.texts)
    )


@app.get("/models", tags=["Model Management"])
async def list_models():
    """List all loaded models and their routing configuration."""
    return {
        "loaded_models": registry.loaded_models,
        "routing_config": registry.routing_config,
        "request_counts": registry.request_counts
    }


@app.put("/models/routing", tags=["Model Management"])
async def update_routing(config: Dict[str, float]):
    """
    Update A/B testing traffic split.
    Example: {"production": 0.8, "staging": 0.2}
    """
    total = sum(config.values())
    if abs(total - 1.0) > 0.01:
        raise HTTPException(status_code=400, detail="Traffic fractions must sum to 1.0")

    for version in config:
        if version not in registry.loaded_models:
            raise HTTPException(
                status_code=400,
                detail=f"Model version '{version}' not loaded"
            )

    registry.routing_config = config
    return {"message": "Routing updated", "config": config}


@app.get("/stats", tags=["Monitoring"])
async def get_stats():
    """Get API usage statistics."""
    return {
        "total_requests": sum(registry.request_counts.values()),
        "requests_by_model": registry.request_counts,
        "models_available": registry.loaded_models,
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        reload=True,
        log_level="info"
    )
