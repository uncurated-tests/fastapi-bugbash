from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


# ---------------------------------------------------------------------------
# Model loading (lazy, cached)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_qa_pipeline():
    """Load the Q&A pipeline once and cache it."""
    from transformers import pipeline

    return pipeline(
        "question-answering",
        model="distilbert-base-uncased-distilled-squad",
        tokenizer="distilbert-base-uncased-distilled-squad",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the model on startup so the first request isn't slow
    get_qa_pipeline()
    yield


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class QARequest(BaseModel):
    question: str
    context: str


class QAResponse(BaseModel):
    answer: str
    score: float
    start: int
    end: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
def read_root():
    return {"message": "Hello, World!", "version": "0.3.0"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask", response_model=QAResponse)
def ask_question(req: QARequest):
    """
    Extractive question-answering.

    Send a **context** paragraph and a **question**; the model extracts
    the most likely answer span from the context.

    Example request body:
    ```json
    {
      "question": "What is the capital of France?",
      "context": "France is a country in Europe. Its capital is Paris."
    }
    ```
    """
    qa = get_qa_pipeline()
    result = qa(question=req.question, context=req.context)
    return QAResponse(
        answer=result["answer"],
        score=round(result["score"], 4),
        start=result["start"],
        end=result["end"],
    )


# ---------------------------------------------------------------------------
# Existing data-science endpoints
# ---------------------------------------------------------------------------


@app.get("/stats")
def compute_stats():
    """Generate random data and return basic statistics."""
    data = np.random.randn(1000)
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "median": float(np.median(data)),
        "skewness": float(stats.skew(data)),
        "kurtosis": float(stats.kurtosis(data)),
    }


@app.get("/regression")
def run_regression():
    """Fit a simple linear regression on random data."""
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    y = 2.5 * X.squeeze() + np.random.randn(100) * 2 + 5

    model = LinearRegression()
    model.fit(X, y)

    return {
        "coefficient": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "r_squared": float(model.score(X, y)),
    }


@app.get("/dataframe")
def dataframe_summary():
    """Create a pandas DataFrame and return a summary."""
    df = pd.DataFrame(
        {
            "a": np.random.randn(500),
            "b": np.random.randint(0, 100, 500),
            "c": np.random.choice(["cat", "dog", "bird"], 500),
        }
    )
    return {
        "shape": list(df.shape),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "describe": df.describe().to_dict(),
    }
