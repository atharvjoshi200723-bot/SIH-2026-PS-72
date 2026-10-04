"""
routes/nowcast.py — Endpoints for triggering nowcast runs and retrieving forecast grids.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, HTTPException

from api.database import save_alert, save_forecast_run
from api.schemas import (
    DistrictImpactModel,
    FrameSummary,
    NowcastRunRequest,
    NowcastRunResponse,
)
from api.services import service
from src.config import cfg

router = APIRouter(prefix="/api/nowcast", tags=["Nowcasting"])


@router.post("/run", response_model=NowcastRunResponse)
async def run_nowcast(req: NowcastRunRequest) -> NowcastRunResponse:
    """Execute a nowcast cycle with real-time sensor health evaluation."""
    input_frames = service.get_sample_input(req.event_idx)
    service.latest_input_frames = input_frames

    # Run inference engine with fallback tier hierarchy
    result = service.inference_engine.predict(
        radar_frames=input_frames,
        radar_age_min=req.radar_age_min,
        satellite_age_min=req.satellite_age_min,
        aws_age_min=req.aws_age_min,
        nwp_age_min=req.nwp_age_min,
        forced_tier=req.forced_tier,
    )
    service.latest_result = result

    # Spatial intersection with administrative districts
    impacts = service.district_join.assess_impacts(result.forecast_grid)
    service.latest_impacts = impacts

    run_id = f"RUN-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:4].upper()}"
    ts = datetime.now(timezone.utc).isoformat()

    # Summaries for response and DB
    frame_summaries = [
        FrameSummary(
            step_index=s.step_index,
            lead_time_min=s.lead_time_min,
            peak_value=s.peak_value,
            mean_value=s.mean_value,
            severe_pixel_fraction=s.severe_pixel_fraction,
            risk_level=s.risk_level,
            risk_colour=s.risk_colour,
        )
        for s in result.risk_by_lead_time
    ]

    district_models = [
        DistrictImpactModel(
            district_id=d.district_id,
            district_name=d.district_name,
            state_name=d.state_name,
            max_risk_level=d.max_risk_level,
            peak_reflectivity=d.peak_reflectivity,
            earliest_arrival_min=d.earliest_arrival_min,
            impacted_area_sq_km=d.impacted_area_sq_km,
            centroid_lat=d.centroid_lat,
            centroid_lon=d.centroid_lon,
        )
        for d in impacts
    ]

    # Save run to SQLite
    save_forecast_run(
        run_id=run_id,
        timestamp=ts,
        tier_used=result.tier_used,
        confidence=result.confidence,
        latency_ms=result.latency_ms,
        max_risk=result.max_overall_risk,
        sensor_status=result.sensor_status,
        risk_summary=[s.model_dump() for s in frame_summaries],
    )

    # Draft alert in PENDING_REVIEW if hazardous
    alert_id = None
    alert_status = None
    if result.max_overall_risk in ["Yellow", "Orange", "Red"] and len(impacts) > 0:
        alert_record = service.alert_workflow.create_alert_from_inference(
            result, impacts
        )
        alert_id = alert_record.alert_id
        alert_status = alert_record.status

        # Persist alert in SQLite
        save_alert(
            alert_id=alert_id,
            status=alert_status,
            created_at=alert_record.created_at,
            overall_risk=alert_record.overall_risk,
            tier_used=alert_record.tier_used,
            confidence=alert_record.confidence,
            districts=[d.model_dump() for d in district_models],
        )

    return NowcastRunResponse(
        run_id=run_id,
        timestamp=ts,
        tier_used=result.tier_used,
        tier_name=result.tier_name,
        tier_description=result.tier_description,
        confidence=result.confidence,
        latency_ms=result.latency_ms,
        max_overall_risk=result.max_overall_risk,
        max_overall_colour=result.max_overall_colour,
        sensor_status=result.sensor_status,
        risk_by_lead_time=frame_summaries,
        impacted_districts=district_models,
        alert_id=alert_id,
        alert_status=alert_status,
    )


@router.get("/latest")
async def get_latest_nowcast() -> dict:
    """Return summary of the most recently executed nowcast."""
    if service.latest_result is None:
        # Automatically run a default nowcast if none cached yet
        req = NowcastRunRequest()
        await run_nowcast(req)

    res = service.latest_result
    assert res is not None
    return {
        "tier_used": res.tier_used,
        "tier_name": res.tier_name,
        "confidence": res.confidence,
        "latency_ms": res.latency_ms,
        "max_overall_risk": res.max_overall_risk,
        "max_overall_colour": res.max_overall_colour,
        "sensor_status": res.sensor_status,
        "risk_by_lead_time": [
            {
                "step_index": s.step_index,
                "lead_time_min": s.lead_time_min,
                "peak_value": s.peak_value,
                "mean_value": s.mean_value,
                "risk_level": s.risk_level,
                "risk_colour": s.risk_colour,
            }
            for s in res.risk_by_lead_time
        ],
        "impacted_districts": [d.to_dict() for d in service.latest_impacts],
    }


@router.get("/frame/{step_idx}")
async def get_forecast_frame(step_idx: int) -> dict:
    """
    Retrieve 2D reflectivity grid for a specific forecast step (0 to 11).
    Used by MapLibre canvas overlay to render radar reflectivity fields.
    """
    if service.latest_result is None:
        await run_nowcast(NowcastRunRequest())

    res = service.latest_result
    assert res is not None

    if step_idx < 0 or step_idx >= res.forecast_grid.shape[0]:
        raise HTTPException(
            status_code=400,
            detail=f"step_idx {step_idx} out of range [0, {res.forecast_grid.shape[0] - 1}]",
        )

    frame = res.forecast_grid[step_idx]
    lead_min = (step_idx + 1) * int(cfg.time.step_minutes)

    return {
        "step_index": step_idx,
        "lead_time_min": lead_min,
        "grid_shape": list(frame.shape),
        "bounds": {
            "lat_min": 19.7,
            "lat_max": 20.9,
            "lon_min": 85.2,
            "lon_max": 86.4,
        },
        "values": np.round(frame, 3).tolist(),
    }


@router.get("/radar-input/{frame_idx}")
async def get_radar_input_frame(frame_idx: int) -> dict:
    """Retrieve historical input radar observation frame (0 to 3)."""
    if service.latest_input_frames is None:
        await run_nowcast(NowcastRunRequest())

    frames = service.latest_input_frames
    assert frames is not None

    if frame_idx < 0 or frame_idx >= frames.shape[0]:
        raise HTTPException(
            status_code=400,
            detail=f"frame_idx {frame_idx} out of range [0, {frames.shape[0] - 1}]",
        )

    frame = frames[frame_idx]
    minutes_ago = (frames.shape[0] - frame_idx - 1) * int(cfg.time.step_minutes)

    return {
        "frame_index": frame_idx,
        "minutes_ago": minutes_ago,
        "grid_shape": list(frame.shape),
        "bounds": {
            "lat_min": 19.7,
            "lat_max": 20.9,
            "lon_min": 85.2,
            "lon_max": 86.4,
        },
        "values": np.round(frame, 3).tolist(),
    }
