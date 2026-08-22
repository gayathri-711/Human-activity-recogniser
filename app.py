from fastapi import FastAPI

app = FastAPI(
    title="AI Human Activity Recognition"
)


@app.get("/")
def home():
    return {
        "status": "success",
        "message": "AI Human Activity Recognition is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }