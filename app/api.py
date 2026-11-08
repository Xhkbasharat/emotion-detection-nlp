
"""
Advanced Emotion Detection REST API
Framework: FastAPI
Model: Fine-tuned DistilBERT
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.predictor import EmotionPredictor


# ==========================================
# 1. LOGGING CONFIGURATION
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

predictor = None


# ==========================================
# 2. APPLICATION LIFESPAN
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global predictor

    logger.info("Loading emotion detection model...")

    predictor = EmotionPredictor()

    logger.info("Model loaded successfully.")

    yield

    predictor = None

    logger.info("API shutting down.")


# ==========================================
# 3. FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="AI Emotion Detection API",
    description=(
        "Advanced NLP emotion classification "
        "using fine-tuned DistilBERT."
    ),
    version="1.0.0",
    lifespan=lifespan
)


# ==========================================
# 4. REQUEST SCHEMA
# ==========================================

class PredictionRequest(BaseModel):

    text: str = Field(
        min_length=1,
        max_length=5000,
        description="Text to analyze for emotional patterns",
        examples=["I am very happy today!"]
    )


# ==========================================
# 5. HEALTH CHECK
# ==========================================

@app.get("/")
async def home():

    return {
        "message": "AI Emotion Detection API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():

    return {
        "status": "healthy" if predictor else "loading",
        "model_loaded": predictor is not None
    }


# ==========================================
# 6. EMOTION PREDICTION ENDPOINT
# ==========================================

@app.post("/predict")
async def predict_emotion(request: PredictionRequest):

    if predictor is None:

        raise HTTPException(
            status_code=503,
            detail="Model is not available."
        )

    try:

        result = predictor.predict(request.text)

        logger.info(
            "Prediction completed: %s",
            result["predicted_emotion"]
        )

        return {
            "success": True,
            "data": result
        }

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception:

        logger.exception("Prediction failed.")

        raise HTTPException(
            status_code=500,
            detail="Internal prediction error."
        )


# ==========================================
# 7. TOP EMOTIONS ENDPOINT
# ==========================================

@app.post("/predict/top")
async def predict_top_emotions(
    request: PredictionRequest
):

    if predictor is None:

        raise HTTPException(
            status_code=503,
            detail="Model is not available."
        )

    try:

        top_results = predictor.top_emotions(
            request.text,
            top_k=3
        )

        return {
            "success": True,
            "top_emotions": [
                {
                    "emotion": emotion,
                    "probability": score
                }
                for emotion, score in top_results
            ]
        }

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception:

        logger.exception("Top emotion prediction failed.")

        raise HTTPException(
            status_code=500,
            detail="Internal prediction error."
  )
      
