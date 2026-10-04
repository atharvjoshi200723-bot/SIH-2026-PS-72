"""
schemas.py — Pydantic request and response models for the VajraDrishti API.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class NowcastRunRequest(BaseModel):
    """Parameters for running a nowcast cycle."""

    radar_age_min: float = Field(
        default=5.0, description="Minutes since last Doppler radar update"
    )
    satellite_age_min: float = Field(
        default=15.0, description="Minutes since last INSAT update"
    )
    aws_age_min: float = Field(
        default=10.0, description="Minutes since last AWS telemetry"
    )
    nwp_age_min: float = Field(
        default=60.0, description="Minutes since last NWP model cycle"
    )
    forced_tier: str | None = Field(
        default=None, description="Optional tier override (T0, T1, T2, T3)"
    )
    event_idx: int | None = Field(
        default=None, description="Index of synthetic event to simulate (0-299)"
    )


class FrameSummary(BaseModel):
    step_index: int
    lead_time_min: int
    peak_value: float
    mean_value: float
    severe_pixel_fraction: float
    risk_level: str
    risk_colour: str


class DistrictImpactModel(BaseModel):
    district_id: str
    district_name: str
    state_name: str
    max_risk_level: str
    peak_reflectivity: float
    earliest_arrival_min: int
    impacted_area_sq_km: int
    centroid_lat: float
    centroid_lon: float


class NowcastRunResponse(BaseModel):
    run_id: str
    timestamp: str
    tier_used: str
    tier_name: str
    tier_description: str
    confidence: float
    latency_ms: float
    max_overall_risk: str
    max_overall_colour: str
    sensor_status: dict[str, Any]
    risk_by_lead_time: list[FrameSummary]
    impacted_districts: list[DistrictImpactModel]
    alert_id: str | None = None
    alert_status: str | None = None


class AlertReviewRequest(BaseModel):
    forecaster_id: str = Field(
        default="duty_meteorologist_01", description="Forecaster identity"
    )
    notes: str | None = Field(
        default=None, description="Meteorological rationale or remarks"
    )
    reason: str | None = Field(
        default=None, description="Rejection reason if applicable"
    )


class AlertResponse(BaseModel):
    alert_id: str
    status: str
    created_at: str
    overall_risk: str
    tier_used: str
    confidence: float
    districts: list[dict[str, Any]]
    forecaster_id: str | None = None
    forecaster_notes: str | None = None
    reviewed_at: str | None = None
    rejection_reason: str | None = None
    has_cap_xml: bool = False
