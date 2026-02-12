# FastAPI Bug Bash

A basic FastAPI hello world application.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

## Endpoints

- `GET /` - Returns a hello world message
- `GET /health` - Health check endpoint
- `GET /docs` - Interactive API documentation (Swagger UI)
