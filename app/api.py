"""
FastAPI Backend Service for Amharic Predictive Text (Q4)
Course: Natural Language Processing | Assignment 1

"""

import os
import sys
from typing import List, Optional
from fastapi import FastAPI, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if os.path.join(ROOT_DIR, "app") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT_DIR, "app"))

from predictor import get_predictor, suggest_candidates

app = FastAPI(
    title="Amharic Predictive Text API",
    description="REST API serving N-gram Next-Word Suggestions on the GPAC Corpus",
    version="1.0.0"
)

# Enable CORS for local Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event: warm up the model
@app.on_event("startup")
def startup_event():
    get_predictor()

# Pydantic Models matching Section 6.2 of UI Plan
class SuggestionItem(BaseModel):
    word: str
    probability: float

class SuggestResponse(BaseModel):
    input_tail: List[str]
    requested_n: int
    model_used: int
    fallback: bool
    strategy: str
    latency_ms: float
    suggestions: List[SuggestionItem]

class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    vocab_size: int

class ErrorResponse(BaseModel):
    error: str
    message: str


@app.get("/api/health", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint confirming API status and model readiness."""
    predictor = get_predictor()
    return {
        "status": "ok",
        "model_ready": predictor.is_ready,
        "vocab_size": predictor.V
    }


@app.get("/suggest", response_model=SuggestResponse)
@app.get("/api/suggest", response_model=SuggestResponse)
def suggest_endpoint(
    text: str = Query(..., description="Amharic text prefix or partial sentence", min_length=1, max_length=500),
    n: int = Query(3, description="N-gram order (2 for Bigram, 3 for Trigram)", ge=2, le=3),
    k: int = Query(5, description="Number of suggestions to return", ge=1, le=5)
):
    """
    Get top-k next word predictions given incomplete Amharic text.
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_input", "message": "Text parameter cannot be empty or solely whitespace."}
        )

    try:
        result = suggest_candidates(cleaned_text, n=n, k=k)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "internal_error", "message": f"Prediction failed: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
