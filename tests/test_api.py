"""
API & Engine Tests for Amharic Predictive Text Backend
Course: Natural Language Processing | Assignment 1 (Q4)
"""

import sys
import os
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if os.path.join(ROOT_DIR, "app") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT_DIR, "app"))

from fastapi.testclient import TestClient
from api import app
from predictor import get_predictor, suggest_candidates

client = TestClient(app)

@pytest.fixture(scope="session", autouse=True)
def warm_up_model():
    """Ensure predictor is loaded before running tests."""
    predictor = get_predictor()
    assert predictor.is_ready, "Model failed to initialize"

def test_health_endpoint():
    """Verify health check endpoint returns 200 and model metadata."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_ready"] is True
    assert data["vocab_size"] > 100000

def test_suggest_example_input():
    """Verify /suggest endpoint on assignment example: 'ኢትዮጵያ ውስጥ የቡና'."""
    text = "ኢትዮጵያ ውስጥ የቡና"
    response = client.get(f"/suggest?text={text}&n=3&k=5")
    assert response.status_code == 200
    data = response.json()
    
    # Contract verification from Section 6.2
    assert "input_tail" in data
    assert "requested_n" in data
    assert "model_used" in data
    assert "fallback" in data
    assert "suggestions" in data
    assert "latency_ms" in data
    
    assert data["requested_n"] == 3
    assert isinstance(data["suggestions"], list)
    assert len(data["suggestions"]) > 0
    assert len(data["suggestions"]) <= 5
    
    # Ensure suggestions are sorted by probability descending
    probs = [item["probability"] for item in data["suggestions"]]
    assert probs == sorted(probs, reverse=True)
    
    for item in data["suggestions"]:
        assert "word" in item
        assert "probability" in item
        assert item["probability"] > 0.0

def test_suggest_bigram():
    """Verify bigram requested_n = 2."""
    text = "ኢትዮጵያ"
    response = client.get(f"/suggest?text={text}&n=2&k=3")
    assert response.status_code == 200
    data = response.json()
    assert data["requested_n"] == 2
    assert len(data["suggestions"]) <= 3

def test_empty_text_returns_error():
    """Verify that an empty or whitespace-only text query returns a 400/422 error."""
    response = client.get("/suggest?text=   &n=3&k=5")
    assert response.status_code in (400, 422)

def test_invalid_n_returns_error():
    """Verify that n < 2 or n > 3 returns validation error."""
    res_low = client.get("/suggest?text=ኢትዮጵያ&n=1&k=5")
    assert res_low.status_code in (400, 422)
    
    res_high = client.get("/suggest?text=ኢትዮጵያ&n=5&k=5")
    assert res_high.status_code in (400, 422)

def test_unseen_context_fallback():
    """Verify that rare/unseen token combinations trigger fallback."""
    unseen_prompt = "ዝሆን አንበሳ ድመት"
    data = suggest_candidates(unseen_prompt, n=3, k=5)
    assert "fallback" in data
    assert isinstance(data["fallback"], bool)
    assert len(data["suggestions"]) > 0
