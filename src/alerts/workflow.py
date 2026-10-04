"""
workflow.py — Human-in-the-Loop (HITL) Forecaster Approval Gate.

A critical design requirement for VajraDrishti:
No public warning or CAP 1.2 XML can be broadcast automatically by AI.
All machine-generated alerts enter a PENDING_REVIEW status and require
explicit human validation by an IMD duty meteorologist.

Forecaster options:
1. Approve: confirms the hazard, optionally adding meteorologist remarks.
   Generates official signed CAP 1.2 XML.
2. Reject: flags false alarms (e.g. radar ground clutter, anomalous propagation).
   Records reason in the audit trail; no public message is dispatched.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from src.alerts.cap_xml import generate_cap_xml
from src.alerts.district_join import DistrictImpact
from src.inference import InferenceResult

logger = logging.getLogger(__name__)


@dataclass
class AlertRecord:
    """Represents a nowcast alert through its lifecycle."""

    alert_id: str
    status: str  # "PENDING_REVIEW" | "APPROVED" | "REJECTED"
    created_at: str
    overall_risk: str  # "Red" | "Orange" | "Yellow"
    tier_used: str
    confidence: float
    impacted_districts: list[DistrictImpact]
    forecaster_id: str | None = None
    forecaster_notes: str | None = None
    reviewed_at: str | None = None
    rejection_reason: str | None = None
    cap_xml: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["impacted_districts"] = [dist.to_dict() for dist in self.impacted_districts]
        return d


class AlertWorkflowManager:
    """
    Manages alert creation, forecaster review queue, and CAP XML generation.
    """

    def __init__(self) -> None:
        self._alerts: dict[str, AlertRecord] = {}

    def create_alert_from_inference(
        self,
        inference_result: InferenceResult,
        impacted_districts: list[DistrictImpact],
    ) -> AlertRecord:
        """
        Draft a new alert from model inference.
        Starts strictly in PENDING_REVIEW state.
        """
        alert_id = f"IN-IMD-VD-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}-{uuid.uuid4().hex[:6].upper()}"
        created_at = datetime.now(timezone.utc).isoformat()

        record = AlertRecord(
            alert_id=alert_id,
            status="PENDING_REVIEW",
            created_at=created_at,
            overall_risk=inference_result.max_overall_risk,
            tier_used=inference_result.tier_used,
            confidence=inference_result.confidence,
            impacted_districts=impacted_districts,
        )
        self._alerts[alert_id] = record
        logger.info(
            "Created alert draft %s (%s) for %d districts. Status: PENDING_REVIEW",
            alert_id,
            record.overall_risk,
            len(impacted_districts),
        )
        return record

    def approve_alert(
        self,
        alert_id: str,
        forecaster_id: str = "forecaster_duty",
        forecaster_notes: str | None = None,
    ) -> AlertRecord:
        """
        Forecaster approval action: transitions status to APPROVED and generates CAP XML.
        """
        if alert_id not in self._alerts:
            raise KeyError(f"Alert {alert_id} not found.")

        record = self._alerts[alert_id]
        if record.status != "PENDING_REVIEW":
            raise ValueError(f"Alert {alert_id} is already in state {record.status}.")

        record.status = "APPROVED"
        record.forecaster_id = forecaster_id
        record.forecaster_notes = forecaster_notes
        record.reviewed_at = datetime.now(timezone.utc).isoformat()

        # Generate official OASIS CAP 1.2 XML
        record.cap_xml = generate_cap_xml(
            alert_id=record.alert_id,
            impacted_districts=record.impacted_districts,
            overall_risk=record.overall_risk,
            tier_used=record.tier_used,
            confidence=record.confidence,
            sender=f"{forecaster_id}@imd.gov.in",
            forecaster_notes=forecaster_notes,
        )

        logger.info(
            "Alert %s APPROVED by %s. CAP XML generated.", alert_id, forecaster_id
        )
        return record

    def reject_alert(
        self,
        alert_id: str,
        forecaster_id: str = "forecaster_duty",
        rejection_reason: str = "False alarm or radar clutter",
    ) -> AlertRecord:
        """
        Forecaster rejection action: logs reason, sets status to REJECTED. No XML is generated.
        """
        if alert_id not in self._alerts:
            raise KeyError(f"Alert {alert_id} not found.")

        record = self._alerts[alert_id]
        if record.status != "PENDING_REVIEW":
            raise ValueError(f"Alert {alert_id} is already in state {record.status}.")

        record.status = "REJECTED"
        record.forecaster_id = forecaster_id
        record.rejection_reason = rejection_reason
        record.reviewed_at = datetime.now(timezone.utc).isoformat()

        logger.info(
            "Alert %s REJECTED by %s. Reason: %s",
            alert_id,
            forecaster_id,
            rejection_reason,
        )
        return record

    def get_alert(self, alert_id: str) -> AlertRecord | None:
        return self._alerts.get(alert_id)

    def list_alerts(self, status: str | None = None) -> list[AlertRecord]:
        if status is None:
            return list(self._alerts.values())
        return [a for a in self._alerts.values() if a.status == status]
