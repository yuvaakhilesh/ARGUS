import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from app.core.config import STATIC_DIR
from app.api.routes import router as api_router

logger = logging.getLogger("surveillance.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.api.routes import cleanup_temporary_files
    cleaned = cleanup_temporary_files(max_age_hours=12.0)
    if cleaned > 0:
        logger.info(f"Cleaned {cleaned} orphaned temporary upload files on startup.")
    logger.info("Eagerly loading VideoMAE model at server startup...")
    from app.core.model import VideoAnomalyDetector
    detector = VideoAnomalyDetector.get_instance()
    detector.load_model()
    yield

app = FastAPI(
    title="ARGUS — Forensic Surveillance Intelligence & Crime Detection",
    description="ARGUS Panoptes-Core: Surveillance Intelligence, Forensic Timeline Analysis, and Object Tracking",
    version="2.4.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def serve_index():
    return FileResponse(STATIC_DIR / "index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
