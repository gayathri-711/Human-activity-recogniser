
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


# ==========================================
# PROJECT PATH
# ==========================================

BASE_DIR = Path(__file__).resolve().parent


# ==========================================
# FASTAPI APP
# ==========================================

app = FastAPI(
    title="AI Human Activity Recognition",
    version="1.0.0"
)


# ==========================================
# STATIC FILES
# ==========================================

app.mount(
    "/static",
    StaticFiles(
        directory=str(BASE_DIR / "static")
    ),
    name="static"
)


# ==========================================
# HTML TEMPLATES
# ==========================================

templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)


# ==========================================
# HOME PAGE
# ==========================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "message": "AI Human Activity Recognition is running"
    }


# ==========================================
# IMAGE ANALYSIS
# ==========================================

@app.post("/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    exercise: str = Form("squat")
):

    image_data = await file.read()

    return {
        "activity": "PERSON DETECTED",
        "exercise": exercise,
        "confidence": 0,
        "repetitions": 0,
        "angle": None,
        "person_detected": True,
        "feedback": "Image received successfully.",
        "file_size": len(image_data)
    }


# ==========================================
# LIVE FRAME ANALYSIS
# ==========================================

@app.post("/analyze-frame")
async def analyze_frame(
    file: UploadFile = File(...),
    exercise: str = Form("squat")
):

    frame_data = await file.read()

    return {
        "activity": "ANALYZING",
        "exercise": exercise,
        "confidence": 0,
        "repetitions": 0,
        "angle": None,
        "person_detected": True,
        "feedback": "Frame received successfully.",
        "file_size": len(frame_data)
    }