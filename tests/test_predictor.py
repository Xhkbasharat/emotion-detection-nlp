
"""
Advanced Automated Tests for Emotion Detection
Framework: Pytest
Model: DistilBERT
CI-compatible testing without model checkpoints
"""

import pytest
import torch

from unittest.mock import Mock
from fastapi.testclient import TestClient

from app.predictor import EmotionPredictor
from app.api import app


# ==========================================
# 1. MOCK PREDICTOR FOR CI
# ==========================================

class FakePredictor:

    def predict(self, text):

        return {
            "text": text,
            "predicted_emotion": "joy",
            "confidence": 0.95,
            "all_emotions": {
                "sadness": 0.01,
                "joy": 0.95,
                "love": 0.01,
                "anger": 0.01,
                "fear": 0.01,
                "surprise": 0.01
            }
        }

    def top_emotions(self, text, top_k=3):

        scores = self.predict(text)["all_emotions"]

        return sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True
        )[:top_k]


# ==========================================
# 2. API TEST FIXTURE
# ==========================================

@pytest.fixture
def client(monkeypatch):

    import app.api as api_module

    # Prevent real model loading during CI tests
    monkeypatch.setattr(
        api_module,
        "EmotionPredictor",
        FakePredictor
    )

    with TestClient(app) as test_client:
        yield test_client


# ==========================================
# 3. INPUT VALIDATION TESTS
# ==========================================

def test_empty_text_validation():

    predictor = EmotionPredictor.__new__(
        EmotionPredictor
    )

    with pytest.raises(ValueError):
        predictor.predict("")


def test_whitespace_validation():

    predictor = EmotionPredictor.__new__(
        EmotionPredictor
    )

    with pytest.raises(ValueError):
        predictor.predict("   ")


def test_invalid_input_type():

    predictor = EmotionPredictor.__new__(
        EmotionPredictor
    )

    with pytest.raises(ValueError):
        predictor.predict(123)


def test_text_length_validation():

    predictor = EmotionPredictor.__new__(
        EmotionPredictor
    )

    with pytest.raises(ValueError):
        predictor.predict("A" * 5001)


# ==========================================
# 4. PREDICTION ENGINE TESTS
# ==========================================

def test_prediction_output_structure():

    predictor = EmotionPredictor.__new__(
        EmotionPredictor
    )

    predictor._validate_text = lambda text: text.strip()
    predictor._tokenize = Mock(return_value={})
    predictor.device = torch.device("cpu")

    predictor.model = Mock()

    predictor.model.config.id2label = {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise"
    }

    predictor.model.return_value.logits = torch.tensor(
        [[0.1, 4.0, 0.2, 0.1, 0.1, 0.1]]
    )

    result = predictor.predict("I am happy")

    assert result["predicted_emotion"] == "joy"
    assert "confidence" in result
    assert "all_emotions" in result
    assert len(result["all_emotions"]) == 6


def test_probability_distribution():

    predictor = EmotionPredictor.__new__(
        EmotionPredictor
    )

    predictor._validate_text = lambda text: text.strip()
    predictor._tokenize = Mock(return_value={})
    predictor.device = torch.device("cpu")

    predictor.model = Mock()

    predictor.model.config.id2label = {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise"
    }

    predictor.model.return_value.logits = torch.tensor(
        [[1.0, 2.0, 3.0, 1.0, 1.0, 1.0]]
    )

    result = predictor.predict("Test sentence")

    total = sum(result["all_emotions"].values())

    assert abs(total - 1.0) < 0.001


def test_top_emotions():

    predictor = EmotionPredictor.__new__(
        EmotionPredictor
    )

    predictor.predict = Mock(
        return_value=FakePredictor().predict("I am happy")
    )

    results = predictor.top_emotions(
        "I am happy",
        top_k=3
    )

    assert len(results) == 3
    assert results[0][0] == "joy"


def test_invalid_top_k():

    predictor = EmotionPredictor.__new__(
        EmotionPredictor
    )

    with pytest.raises(ValueError):
        predictor.top_emotions(
            "I am happy",
            top_k=0
        )


# ==========================================
# 5. API ENDPOINT TESTS
# ==========================================

def test_home_endpoint(client):

    response = client.get("/")

    assert response.status_code == 200

    assert response.json()["status"] == "running"


def test_health_endpoint(client):

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json()["model_loaded"] is True


def test_invalid_api_request(client):

    response = client.post(
        "/predict",
        json={"text": ""}
    )

    assert response.status_code == 422


def test_missing_text_api(client):

    response = client.post(
        "/predict",
        json={}
    )

    assert response.status_code == 422


# ==========================================
# 6. API PREDICTION TEST
# ==========================================

def test_prediction_api(client):

    response = client.post(
        "/predict",
        json={"text": "I am happy"}
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True

    assert data["data"]["predicted_emotion"] == "joy"


# ==========================================
# 7. TOP EMOTIONS API TEST
# ==========================================

def test_top_emotions_api(client):

    response = client.post(
        "/predict/top",
        json={"text": "I am very happy"}
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True

    assert len(data["top_emotions"]) == 3

    assert data["top_emotions"][0]["emotion"] == "joy"
    
