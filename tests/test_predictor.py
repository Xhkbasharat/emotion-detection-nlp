
"""
Automated Tests for Emotion Detection
Framework: Pytest
"""

import pytest
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.predictor import EmotionPredictor
from app.api import app


# ==========================================
# 1. INPUT VALIDATION TESTS
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
# 2. PREDICTION OUTPUT TESTS
# ==========================================

def test_prediction_output_structure():

    predictor = EmotionPredictor.__new__(
        EmotionPredictor
    )

    predictor._validate_text = lambda text: text.strip()

    predictor._tokenize = Mock(return_value={})

    predictor.device = "cpu"

    predictor.model = Mock()

    predictor.model.config.id2label = {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise"
    }

    import torch

    predictor.model.return_value.logits = torch.tensor(
        [[0.1, 4.0, 0.2, 0.1, 0.1, 0.1]]
    )

    result = predictor.predict("I am happy")

    assert "text" in result
    assert "predicted_emotion" in result
    assert "confidence" in result
    assert "all_emotions" in result

    assert result["predicted_emotion"] == "joy"


def test_probability_distribution():

    predictor = EmotionPredictor.__new__(
        EmotionPredictor
    )

    predictor._validate_text = lambda text: text.strip()

    predictor._tokenize = Mock(return_value={})

    predictor.device = "cpu"

    predictor.model = Mock()

    predictor.model.config.id2label = {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise"
    }

    import torch

    predictor.model.return_value.logits = torch.tensor(
        [[1.0, 2.0, 3.0, 1.0, 1.0, 1.0]]
    )

    result = predictor.predict("Test sentence")

    total = sum(
        result["all_emotions"].values()
    )

    assert abs(total - 1.0) < 0.001


# ==========================================
# 3. TOP EMOTION TESTS
# ==========================================

def test_top_emotions():

    predictor = EmotionPredictor.__new__(
        EmotionPredictor
    )

    predictor.predict = Mock(
        return_value={
            "text": "I am happy",
            "predicted_emotion": "joy",
            "confidence": 0.8,
            "all_emotions": {
                "joy": 0.8,
                "love": 0.1,
                "surprise": 0.05,
                "sadness": 0.02,
                "anger": 0.02,
                "fear": 0.01
            }
        }
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
# 4. API TESTS
# ==========================================

@pytest.fixture
def client():

    with TestClient(app) as test_client:

        yield test_client


def test_home_endpoint(client):

    response = client.get("/")

    assert response.status_code == 200

    assert (
        response.json()["status"] == "running"
    )


def test_health_endpoint(client):

    response = client.get("/health")

    assert response.status_code == 200

    assert "model_loaded" in response.json()


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
# 5. API MOCK PREDICTION TEST
# ==========================================

def test_prediction_api(client, monkeypatch):

    import app.api as api_module

    mock_predictor = Mock()

    mock_predictor.predict.return_value = {
        "text": "I am happy",
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

    monkeypatch.setattr(
        api_module,
        "predictor",
        mock_predictor
    )

    response = client.post(
        "/predict",
        json={"text": "I am happy"}
    )

    assert response.status_code == 200

    assert response.json()["success"] is True

    assert (
        response.json()["data"]["predicted_emotion"]
        == "joy"
  )
  
