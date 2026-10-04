"""
routes/alerts.py — Human-in-the-Loop Forecaster Approval Gate endpoints & CAP 1.2 XML download.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, Response

from api.database import get_alert, list_alerts, log_audit, update_alert
from api.schemas import AlertResponse, AlertReviewRequest
from api.services import service

router = APIRouter(prefix="/api/alerts", tags=["Alerts"])


@router.get("", response_model=list[AlertResponse])
async def get_all_alerts(
    status: str | None = Query(default=None),
) -> list[AlertResponse]:
    """Retrieve alerts, optionally filtered by status ('PENDING_REVIEW', 'APPROVED', 'REJECTED')."""
    records = list_alerts(status=status)
    return [
        AlertResponse(
            alert_id=r["alert_id"],
            status=r["status"],
            created_at=r["created_at"],
            overall_risk=r["overall_risk"],
            tier_used=r["tier_used"],
            confidence=r["confidence"],
            districts=r["districts"],
            forecaster_id=r.get("forecaster_id"),
            forecaster_notes=r.get("forecaster_notes"),
            reviewed_at=r.get("reviewed_at"),
            rejection_reason=r.get("rejection_reason"),
            has_cap_xml=bool(r.get("cap_xml")),
        )
        for r in records
    ]


@router.get("/pending", response_model=list[AlertResponse])
async def get_pending_alerts() -> list[AlertResponse]:
    """Retrieve alerts requiring urgent forecaster approval."""
    return await get_all_alerts(status="PENDING_REVIEW")


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_single_alert(alert_id: str) -> AlertResponse:
    """Retrieve details for a specific alert."""
    r = get_alert(alert_id)
    if not r:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found.")

    return AlertResponse(
        alert_id=r["alert_id"],
        status=r["status"],
        created_at=r["created_at"],
        overall_risk=r["overall_risk"],
        tier_used=r["tier_used"],
        confidence=r["confidence"],
        districts=r["districts"],
        forecaster_id=r.get("forecaster_id"),
        forecaster_notes=r.get("forecaster_notes"),
        reviewed_at=r.get("reviewed_at"),
        rejection_reason=r.get("rejection_reason"),
        has_cap_xml=bool(r.get("cap_xml")),
    )


@router.post("/{alert_id}/approve", response_model=AlertResponse)
async def approve_alert_endpoint(
    alert_id: str, review: AlertReviewRequest
) -> AlertResponse:
    """
    Forecaster approval gate: Approves a pending alert and generates signed OASIS CAP 1.2 XML.
    """
    r = get_alert(alert_id)
    if not r:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found.")
    if r["status"] != "PENDING_REVIEW":
        raise HTTPException(
            status_code=400, detail=f"Alert is already in state '{r['status']}'."
        )

    # Call workflow manager to approve and produce CAP XML
    # If workflow manager doesn't have it in memory (e.g. server restart), register it first
    wf = service.alert_workflow
    if wf.get_alert(alert_id) is None:
        from src.alerts.district_join import DistrictImpact
        from src.alerts.workflow import AlertRecord

        impact_objs = [
            DistrictImpact(
                district_id=d.get("district_id", "OD_XX"),
                district_name=d.get("district_name", "Unknown"),
                state_name=d.get("state_name", "Odisha"),
                max_risk_level=d.get("max_risk_level", "Yellow"),
                peak_reflectivity=float(d.get("peak_reflectivity", 0.5)),
                earliest_arrival_min=int(d.get("earliest_arrival_min", 15)),
                impacted_area_sq_km=int(d.get("impacted_area_sq_km", 10)),
                centroid_lat=float(d.get("centroid_lat", 20.0)),
                centroid_lon=float(d.get("centroid_lon", 85.5)),
            )
            for d in r["districts"]
        ]
        wf._alerts[alert_id] = AlertRecord(
            alert_id=alert_id,
            status=r["status"],
            created_at=r["created_at"],
            overall_risk=r["overall_risk"],
            tier_used=r["tier_used"],
            confidence=r["confidence"],
            impacted_districts=impact_objs,
        )

    approved_record = wf.approve_alert(
        alert_id=alert_id,
        forecaster_id=review.forecaster_id,
        forecaster_notes=review.notes,
    )

    now_ts = datetime.now(timezone.utc).isoformat()
    # Update SQLite record
    update_alert(
        alert_id=alert_id,
        status="APPROVED",
        forecaster_id=review.forecaster_id,
        reviewed_at=now_ts,
        notes=review.notes,
        cap_xml=approved_record.cap_xml,
    )

    # Log audit entry
    log_audit(
        action="ALERT_APPROVED",
        actor=review.forecaster_id,
        details=f"Alert {alert_id} ({r['overall_risk']}) approved. CAP XML generated.",
        timestamp=now_ts,
    )

    return await get_single_alert(alert_id)


@router.post("/{alert_id}/reject", response_model=AlertResponse)
async def reject_alert_endpoint(
    alert_id: str, review: AlertReviewRequest
) -> AlertResponse:
    """
    Forecaster rejection action: Suppresses alert broadcast with an audited rationale.
    """
    r = get_alert(alert_id)
    if not r:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found.")
    if r["status"] != "PENDING_REVIEW":
        raise HTTPException(
            status_code=400, detail=f"Alert is already in state '{r['status']}'."
        )

    now_ts = datetime.now(timezone.utc).isoformat()
    reason = review.reason or review.notes or "False alarm flagged by forecaster"

    update_alert(
        alert_id=alert_id,
        status="REJECTED",
        forecaster_id=review.forecaster_id,
        reviewed_at=now_ts,
        rejection_reason=reason,
    )

    log_audit(
        action="ALERT_REJECTED",
        actor=review.forecaster_id,
        details=f"Alert {alert_id} rejected. Reason: {reason}",
        timestamp=now_ts,
    )

    return await get_single_alert(alert_id)


@router.get("/{alert_id}/cap.xml")
async def download_cap_xml(alert_id: str) -> Response:
    """Download official OASIS CAP 1.2 XML document for an approved alert."""
    r = get_alert(alert_id)
    if not r:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found.")
    if not r.get("cap_xml"):
        raise HTTPException(
            status_code=400,
            detail=f"Alert {alert_id} is in status '{r['status']}' and has no approved CAP XML.",
        )

    return Response(
        content=r["cap_xml"],
        media_type="application/xml",
        headers={
            "Content-Disposition": f"attachment; filename=cap_alert_{alert_id}.xml",
        },
    )
