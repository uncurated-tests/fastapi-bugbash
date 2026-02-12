from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello, World!", "version": "0.1.0"}


@app.get("/health")
def health_check():
    return {"status": "ok"}
