"""FastAPI application entry point."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from contextlib import asynccontextmanager
import logging

from api.config import settings
from api.routers import verification, health, admin, websocket
from api.services.pipeline_manager import VerificationPipeline
from api.models.database import engine, Base

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting up Retin-Verify API...")
    
    # Create database tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created")
    
    # Initialize pipeline
    app.state.pipeline = VerificationPipeline()
    logger.info("Pipeline initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Retin-Verify API...")


# Create FastAPI application
app = FastAPI(
    title="Retin-Verify V3 API",
    description="Computer Vision-Based Algerian ID Card Verification",
    version="3.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(verification.router, prefix="/api/v1", tags=["verification"])
app.include_router(health.router, prefix="", tags=["health"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])
app.include_router(websocket.router, prefix="/ws", tags=["websocket"])

# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint - serves web UI."""
    index_path = static_path / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    return JSONResponse({
        "message": "Retin-Verify V3 API",
        "version": "3.0.0",
        "docs": "/docs"
    })


@app.get("/api")
async def api_info():
    """API information."""
    return {
        "name": "Retin-Verify V3 API",
        "version": "3.0.0",
        "endpoints": {
            "verification": "/api/v1/verify",
            "health": "/health",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS if not settings.DEBUG else 1
    )
