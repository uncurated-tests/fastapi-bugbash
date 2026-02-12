from fastapi import FastAPI
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from PIL import Image

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello, World!", "version": "0.2.0"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


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
