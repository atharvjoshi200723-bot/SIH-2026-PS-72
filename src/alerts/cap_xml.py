"""
cap_xml.py — OASIS Common Alerting Protocol (CAP) v1.2 XML Generator.

Generates machine-readable, standard-compliant CAP 1.2 XML alert messages
for severe thunderstorm and lightning nowcasting, compatible with disaster
management pipelines (NDMA SACHET / State SDMAs).

Standard Reference:
    OASIS Standard CAP-V1.2, 01 July 2010.
    http://docs.oasis-open.org/emergency/cap/v1.2/CAP-v1.2.html
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

from src.alerts.district_join import DistrictImpact

CAP_NAMESPACE = "urn:oasis:names:tc:emergency:cap:1.2"
IST = timezone(timedelta(hours=5, minutes=30))


def format_iso_timestamp(dt: datetime | None = None) -> str:
    """Format datetime in ISO 8601 with Indian Standard Time (+05:30) offset."""
    if dt is None:
        dt = datetime.now(IST)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)
    return dt.isoformat(timespec="seconds")


def get_cap_severity(imd_risk: str) -> str:
    """Map IMD warning colors to OASIS CAP 1.2 severity values."""
    mapping = {
        "Red": "Extreme",
        "Orange": "Severe",
        "Yellow": "Moderate",
        "Green": "Minor",
    }
    return mapping.get(imd_risk, "Moderate")


def get_cap_urgency(earliest_arrival_min: int) -> str:
    """Map lead time to CAP urgency."""
    if earliest_arrival_min <= 60:
        return "Immediate"
    elif earliest_arrival_min <= 120:
        return "Expected"
    return "Future"


def get_public_safety_instructions(risk_level: str) -> str:
    """Actionable lightning and thunderstorm safety guidance."""
    if risk_level in ["Red", "Orange"]:
        return (
            "1. Farmers and outdoor workers must immediately suspend fieldwork and seek sturdy indoor shelter. "
            "2. Do NOT take shelter under isolated tall trees or tin sheds. "
            "3. Keep away from metal fences, tractors, electric lines, and open water bodies. "
            "4. If caught in an open field with no shelter nearby, crouch low on the balls of your feet with hands on knees; do NOT lie flat. "
            "5. Unplug sensitive electrical appliances and avoid using corded phones."
        )
    return (
        "1. Monitor local weather bulletins and radar updates. "
        "2. Avoid venturing out during thunderstorm activity unless necessary. "
        "3. Secure loose outdoor objects."
    )


def generate_cap_xml(
    alert_id: str,
    impacted_districts: list[DistrictImpact],
    overall_risk: str,
    tier_used: str,
    confidence: float,
    sender: str = "duty.forecaster@imd.gov.in",
    sent_time: datetime | None = None,
    expiry_hours: float = 3.0,
    forecaster_notes: str | None = None,
) -> str:
    """
    Construct OASIS CAP 1.2 XML document string.

    Args:
        alert_id: Unique alert identifier (e.g. IN-IMD-VD-20261004-001)
        impacted_districts: List of DistrictImpact objects from spatial join
        overall_risk: "Red" | "Orange" | "Yellow"
        tier_used: Sensor fallback tier ("T0", "T1", "T2", "T3")
        confidence: Model confidence score in [0, 1]
        sender: Sending authority email/URI
        sent_time: Sent datetime (defaults to now in IST)
        expiry_hours: Validity duration (default: 3 hours)
        forecaster_notes: Optional remarks from reviewing meteorologist

    Returns:
        Formatted XML string conforming to CAP 1.2 schema.
    """
    now = sent_time or datetime.now(IST)
    expires = now + timedelta(hours=expiry_hours)

    sent_str = format_iso_timestamp(now)
    expires_str = format_iso_timestamp(expires)

    cap_severity = get_cap_severity(overall_risk)
    earliest_min = min((d.earliest_arrival_min for d in impacted_districts), default=15)
    cap_urgency = get_cap_urgency(earliest_min)

    district_names = [d.district_name for d in impacted_districts]
    districts_str = ", ".join(district_names) if district_names else "Specified Regions"

    headline = (
        f"IMD VajraDrishti {overall_risk} Warning: Severe Thunderstorm & Lightning "
        f"expected in {districts_str} within {earliest_min} minutes"
    )

    desc_parts = [
        "AI Nowcast indicates active convective thunderstorm development with peak reflectivity above threshold.",
        f"Affected Districts: {districts_str}.",
        f"Operational Sensor Tier: {tier_used} (Model Confidence: {confidence * 100:.1f}%).",
    ]
    if forecaster_notes:
        desc_parts.append(f"Forecaster Evaluation Notes: {forecaster_notes}")
    description = " ".join(desc_parts)

    instruction = get_public_safety_instructions(overall_risk)

    # ── Build XML tree ────────────────────────────────────────────────────────
    root = ET.Element("alert", xmlns=CAP_NAMESPACE)

    ET.SubElement(root, "identifier").text = alert_id
    ET.SubElement(root, "sender").text = sender
    ET.SubElement(root, "sent").text = sent_str
    ET.SubElement(root, "status").text = "Actual"
    ET.SubElement(root, "msgType").text = "Alert"
    ET.SubElement(root, "scope").text = "Public"

    info = ET.SubElement(root, "info")
    ET.SubElement(info, "language").text = "en-IN"
    ET.SubElement(info, "category").text = "Met"
    ET.SubElement(info, "event").text = "Thunderstorm / Lightning Nowcast Warning"
    ET.SubElement(info, "urgency").text = cap_urgency
    ET.SubElement(info, "severity").text = cap_severity
    ET.SubElement(info, "certainty").text = (
        "Observed" if tier_used == "T0" else "Likely"
    )

    # Event code
    event_code = ET.SubElement(info, "eventCode")
    ET.SubElement(event_code, "valueName").text = "SAME"
    ET.SubElement(event_code, "value").text = "SVR"

    ET.SubElement(info, "expires").text = expires_str
    ET.SubElement(
        info, "senderName"
    ).text = "India Meteorological Department (IMD) - VajraDrishti AI"
    ET.SubElement(info, "headline").text = headline
    ET.SubElement(info, "description").text = description
    ET.SubElement(info, "instruction").text = instruction
    ET.SubElement(
        info, "contact"
    ).text = "National Weather Forecasting Centre, New Delhi"

    # Parameters (IMD Color code, fallback tier, etc.)
    p1 = ET.SubElement(info, "parameter")
    ET.SubElement(p1, "valueName").text = "IMDColorCode"
    ET.SubElement(p1, "value").text = overall_risk

    p2 = ET.SubElement(info, "parameter")
    ET.SubElement(p2, "valueName").text = "FallbackTier"
    ET.SubElement(p2, "value").text = tier_used

    p3 = ET.SubElement(info, "parameter")
    ET.SubElement(p3, "valueName").text = "LeadTimeMinutes"
    ET.SubElement(p3, "value").text = str(earliest_min)

    # Area element
    area = ET.SubElement(info, "area")
    ET.SubElement(area, "areaDesc").text = districts_str

    for d in impacted_districts:
        circle = ET.SubElement(area, "circle")
        # Approximate 10km circle around centroid
        circle.text = f"{d.centroid_lat},{d.centroid_lon} 10.0"

    xml_str = ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8")
    return xml_str


def validate_cap_xml(xml_content: str) -> bool:
    """
    Validate that xml_content is well-formed XML and conforms to CAP 1.2 root tag and namespace.
    """
    try:
        root = ET.fromstring(xml_content.encode("utf-8"))
        # Verify tag and namespace
        expected_tag = f"{{{CAP_NAMESPACE}}}alert"
        if root.tag != expected_tag and root.tag != "alert":
            return False
        # Check required fields
        required_children = [
            "identifier",
            "sender",
            "sent",
            "status",
            "msgType",
            "scope",
            "info",
        ]
        tags = [c.tag.split("}")[-1] for c in root]
        return all(r in tags for r in required_children)
    except ET.ParseError:
        return False
