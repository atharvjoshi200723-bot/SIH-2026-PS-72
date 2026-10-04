"""
routes/system.py — System health, evaluation metrics, district GeoJSON, and audit trail.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.database import get_audit_logs
from api.services import service
from src.config import cfg

router = APIRouter(prefix="/api", tags=["System"])


@router.get("/health")
def health_check() -> dict:
    """Comprehensive health check of system services."""
    model_loaded = service.inference_engine.model is not None
    db_ok = True
    try:
        from api.database import get_db_path

        get_db_path()
    except (OSError, RuntimeError):
        db_ok = False

    return {
        "status": "healthy" if (model_loaded and db_ok) else "degraded",
        "system": cfg.project.name,
        "version": cfg.project.version,
        "ps_id": cfg.project.ps_id,
        "ai_model_loaded": model_loaded,
        "database_connected": db_ok,
        "device": str(service.device),
        "available_tiers": ["T0", "T1", "T2", "T3"],
    }


@router.get("/metrics")
def get_evaluation_metrics() -> dict:
    """Retrieve verified meteorological validation metrics computed in Phase 4."""
    path = Path(cfg.paths.results) / "metrics.json"
    if not path.exists():
        return {
            "error": "metrics.json not yet generated. Run Phase 4 evaluation first."
        }

    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/districts")
def get_districts_geojson() -> JSONResponse:
    """Retrieve sample administrative district GeoJSON with bounding metadata."""
    path = Path(cfg.paths.district_geojson)
    if not path.exists():
        return JSONResponse(
            status_code=404, content={"error": "Districts GeoJSON not found."}
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    return JSONResponse(content=data)


@router.get("/audit-logs")
async def get_system_audit_logs() -> list[dict]:
    """Retrieve recent forecaster actions and approval logs."""
    return get_audit_logs(limit=50)
