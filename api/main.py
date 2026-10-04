"""
main.py — VajraDrishti FastAPI Application Entry Point.

Fuses Doppler radar, satellite imagery, AWS stations, and NWP to provide
1 km x 15-minute thunderstorm and lightning nowcasts (0-180 min) with
a Human-in-the-Loop Forecaster Approval Gate and OASIS CAP 1.2 XML output.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.database import init_db
from api.routes import alerts, nowcast, system


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure SQLite database schema exists
    init_db()
    yield
    # Shutdown cleanups if needed


app = FastAPI(
    title="VajraDrishti ⚡ AI Nowcasting System",
    description=(
        "AI-powered nowcasting of thunderstorms and lightning risk up to 3 hours ahead (15-min steps, 1 km grid) "
        "with Human-in-the-Loop Forecaster Approval Gate and OASIS CAP 1.2 XML output for SIH 2026 PS 26072."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for local frontend dashboard communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include route modules
app.include_router(system.router)
app.include_router(nowcast.router)
app.include_router(alerts.router)


@app.get("/health", tags=["System"])
async def root_health() -> dict:
    """Liveness check for container orchestration and Phase 0 compatibility."""
    return {"status": "ok", "message": "VajraDrishti API is healthy and operational."}


# Mount results directory for skill scores plot
results_dir = Path(__file__).parent.parent / "results"
if results_dir.exists():
    app.mount("/results", StaticFiles(directory=str(results_dir)), name="results")

# Mount web dashboard static files
web_dir = Path(__file__).parent.parent / "web"
if web_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
