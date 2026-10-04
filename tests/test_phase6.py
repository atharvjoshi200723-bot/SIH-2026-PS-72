"""
test_phase6.py — Verification tests for Phase 6 (CAP 1.2 XML, District Spatial Join, Forecaster Approval).

Tests:
1. Spatial intersection between forecast grid and district GeoJSON
2. Correct district identification and impact summary generation
3. OASIS CAP 1.2 XML generation and schema validation
4. Forecaster Approval Gate lifecycle (PENDING_REVIEW -> APPROVED / REJECTED)
5. Audit trail integrity on approved and rejected alerts
"""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from src.alerts.cap_xml import (
    CAP_NAMESPACE,
    generate_cap_xml,
    validate_cap_xml,
)
from src.alerts.district_join import DistrictImpact, DistrictJoinEngine
from src.alerts.workflow import AlertWorkflowManager
from src.config import load_config
from src.inference import InferenceEngine


@pytest.fixture
def cfg():
    return load_config()


def test_district_join_engine(cfg):
    joiner = DistrictJoinEngine(config_obj=cfg)
    assert len(joiner.districts) >= 4

    # Create synthetic forecast with storm cell in center of grid
    T_out = cfg.time.output_frames
    H = cfg.grid.height
    W = cfg.grid.width

    forecast = np.zeros((T_out, H, W), dtype=np.float32)
    # Put intense cell (0.85 -> Red alert) in central quadrant
    forecast[0:3, 40:65, 45:70] = 0.85

    impacts = joiner.assess_impacts(forecast, threshold=0.3)
    assert len(impacts) > 0

    first = impacts[0]
    assert isinstance(first, DistrictImpact)
    assert first.max_risk_level in ["Orange", "Red"]
    assert first.peak_reflectivity >= 0.8
    assert first.earliest_arrival_min >= 15
    assert first.impacted_area_sq_km > 0


def test_cap_xml_generation_and_validation():
    sample_impacts = [
        DistrictImpact(
            district_id="OD_01",
            district_name="Cuttack",
            state_name="Odisha",
            max_risk_level="Red",
            peak_reflectivity=0.88,
            earliest_arrival_min=30,
            impacted_area_sq_km=45,
            centroid_lat=20.46,
            centroid_lon=85.88,
        ),
        DistrictImpact(
            district_id="OD_02",
            district_name="Khordha",
            state_name="Odisha",
            max_risk_level="Orange",
            peak_reflectivity=0.72,
            earliest_arrival_min=45,
            impacted_area_sq_km=28,
            centroid_lat=20.18,
            centroid_lon=85.75,
        ),
    ]

    xml_str = generate_cap_xml(
        alert_id="IN-IMD-VD-20261004-TEST01",
        impacted_districts=sample_impacts,
        overall_risk="Red",
        tier_used="T0",
        confidence=0.92,
        sender="duty.forecaster@imd.gov.in",
        forecaster_notes="Confirmed by Cuttack Doppler Radar high Z reflectivity core.",
    )

    # 1. Structural validity check
    assert validate_cap_xml(xml_str) is True

    # 2. Check XML elements
    root = ET.fromstring(xml_str.encode("utf-8"))
    assert root.tag == f"{{{CAP_NAMESPACE}}}alert"

    info = root.find(f"{{{CAP_NAMESPACE}}}info")
    assert info is not None
    assert info.find(f"{{{CAP_NAMESPACE}}}severity").text == "Extreme"
    assert info.find(f"{{{CAP_NAMESPACE}}}urgency").text == "Immediate"
    assert "Cuttack" in info.find(f"{{{CAP_NAMESPACE}}}headline").text

    # Parameter check
    params = info.findall(f"{{{CAP_NAMESPACE}}}parameter")
    param_dict = {
        p.find(f"{{{CAP_NAMESPACE}}}valueName").text: p.find(
            f"{{{CAP_NAMESPACE}}}value"
        ).text
        for p in params
    }
    assert param_dict.get("IMDColorCode") == "Red"
    assert param_dict.get("FallbackTier") == "T0"


def test_forecaster_approval_workflow(cfg):
    manager = AlertWorkflowManager()
    engine = InferenceEngine(config_obj=cfg)

    # Simulate inference run
    dummy_input = np.zeros(
        (cfg.time.input_frames, cfg.grid.height, cfg.grid.width), dtype=np.float32
    )
    dummy_input[:, 40:65, 45:70] = 0.85
    inf_res = engine.predict(dummy_input)

    joiner = DistrictJoinEngine(config_obj=cfg)
    impacts = joiner.assess_impacts(inf_res.forecast_grid)

    # 1. Create alert draft
    alert = manager.create_alert_from_inference(inf_res, impacts)
    assert alert.status == "PENDING_REVIEW"
    assert alert.cap_xml is None

    # 2. Approve alert
    approved = manager.approve_alert(
        alert.alert_id,
        forecaster_id="imd_scientist_bhubaneswar",
        forecaster_notes="Radar convective signature verified.",
    )
    assert approved.status == "APPROVED"
    assert approved.forecaster_id == "imd_scientist_bhubaneswar"
    assert approved.cap_xml is not None
    assert validate_cap_xml(approved.cap_xml) is True

    # 3. Reject another alert
    alert2 = manager.create_alert_from_inference(inf_res, impacts)
    rejected = manager.reject_alert(
        alert2.alert_id,
        forecaster_id="imd_scientist_bhubaneswar",
        rejection_reason="Sea breeze front reflection, not deep convection.",
    )
    assert rejected.status == "REJECTED"
    assert rejected.cap_xml is None
    assert "Sea breeze" in rejected.rejection_reason
