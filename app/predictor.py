"""
Emotion Detection Prediction Engine
Model: Fine-tuned DistilBERT
Task: Six-class emotion classification
"""

import torch
import logging

from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# ==========================================
# 1. CONFIGURATION
# ==========================================

MODEL_PATH = Path("models/emotion-distilbert")

MAX_LENGTH = 128

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ==========================================
# 2. EMOTION PREDICTOR CLASS
# ==========================================

class EmotionPredictor:

    def __init__(self, model_path=MODEL_PATH):

        self.model_path = Path(model_path)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        logger.info(
            f"Loading model on {self.device}"
        )

        self._load_model()


    # ======================================
    # 3. LOAD TOKENIZER AND MODEL
    # ======================================

    def _load_model(self):

        if not self.model_path.exists():

            raise FileNotFoundError(
                f"Model not found: {self.model_path}. "
                "Train the model or download its checkpoint."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path)
        )

        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                str(self.model_path)
            )
        )

        self.model.to(self.device)

        self.model.eval()

        logger.info("Model loaded successfully")


    # ======================================
    # 4. TEXT VALIDATION
    # ======================================

    def _validate_text(self, text):

        if not isinstance(text, str):

            raise ValueError(
                "Input must be a string."
            )

        if not text.strip():

            raise ValueError(
                "Input text cannot be empty."
            )

        if len(text) > 5000:

            raise ValueError(
                "Input text exceeds 5000 characters."
            )

        return text.strip()


    # ======================================
    # 5. TOKENIZATION
    # ======================================

    def _tokenize(self, text):

        inputs = self.tokenizer(

            text,

            return_tensors="pt",

            truncation=True,

            padding=True,

            max_length=MAX_LENGTH
        )

        return {
            key: value.to(self.device)
            for key, value in inputs.items()
        }


    # ======================================
    # 6. EMOTION PREDICTION
    # ======================================

    def predict(self, text):

        text = self._validate_text(text)

        inputs = self._tokenize(text)

        with torch.inference_mode():

            outputs = self.model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1
            )[0]

        scores = {}

        for index, probability in enumerate(probabilities):

            emotion = self.model.config.id2label.get(
                index,
                str(index)
            )

            scores[emotion] = round(
                float(probability.cpu()),
                4
            )

        predicted_emotion = max(
            scores,
            key=scores.get
        )

        confidence = scores[predicted_emotion]

        return {

            "text": text,

            "predicted_emotion": predicted_emotion,

            "confidence": confidence,

            "all_emotions": scores
        }


    # ======================================
    # 7. TOP EMOTION PREDICTIONS
    # ======================================

    def top_emotions(self, text, top_k=3):

        if not isinstance(top_k, int) or top_k < 1:

            raise ValueError(
                "top_k must be a positive integer."
            )

        result = self.predict(text)

        sorted_emotions = sorted(

            result["all_emotions"].items(),

            key=lambda item: item[1],

            reverse=True
        )

        return sorted_emotions[:top_k]


# ==========================================
# 8. LOCAL TESTING
# ==========================================

if __name__ == "__main__":

    predictor = EmotionPredictor()

    sample_text = (
        "I am extremely happy because "
        "I achieved my dream!"
    )

    result = predictor.predict(sample_text)

    print("\nEMOTION DETECTION RESULT")

    print("------------------------")

    print(
        "Text:",
        result["text"]
    )

    print(
        "Predicted Emotion:",
        result["predicted_emotion"]
    )

    print(
        "Confidence:",
        result["confidence"]
    )

    print("\nAll Emotion Scores:")

    for emotion, score in result["all_emotions"].items():

        print(f"{emotion}: {score}")

    print("\nTop 3 Emotions:")

    print(
        predictor.top_emotions(sample_text, 3)
            )
