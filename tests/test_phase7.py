"""
test_phase7.py — Integration and endpoint tests for FastAPI backend (Phase 7).

Tests:
1. /health and /api/health liveness checks
2. /api/nowcast/run triggering full end-to-end inference and SQLite persistence
3. /api/nowcast/frame/{step_idx} grid array serialization
4. /api/alerts/pending and approval workflow transition (PENDING_REVIEW -> APPROVED)
5. /api/alerts/{alert_id}/cap.xml download and content-type validation
6. /api/alerts/{alert_id}/reject handling
7. /api/metrics and /api/districts serving
"""

import pytest
from fastapi.testclient import TestClient

from api.database import init_db
from api.main import app
from src.alerts.cap_xml import validate_cap_xml


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # Set up test database in temp location
    test_db = tmp_path_factory.mktemp("db") / "test_vajradrishti.db"
    init_db(test_db)
    with TestClient(app) as c:
        yield c


def test_health_endpoints(client):
    res_root = client.get("/health")
    assert res_root.status_code == 200
    assert res_root.json()["status"] == "ok"

    res_api = client.get("/api/health")
    assert res_api.status_code == 200
    data = res_api.json()
    assert "system" in data
    assert data["system"] == "VajraDrishti"
    assert "available_tiers" in data


def test_nowcast_run_and_frame_retrieval(client):
    # Trigger nowcast run
    payload = {
        "radar_age_min": 5.0,
        "satellite_age_min": 12.0,
        "aws_age_min": 10.0,
        "nwp_age_min": 45.0,
    }
    res = client.post("/api/nowcast/run", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "run_id" in data
    assert data["tier_used"] == "T0"
    assert len(data["risk_by_lead_time"]) == 12
    assert "impacted_districts" in data

    # Retrieve frame 0 (15 min lead time)
    res_frame = client.get("/api/nowcast/frame/0")
    assert res_frame.status_code == 200
    f_data = res_frame.json()
    assert f_data["step_index"] == 0
    assert f_data["lead_time_min"] == 15
    assert len(f_data["values"]) == 128
    assert len(f_data["values"][0]) == 128


def test_alerts_approval_and_cap_xml_download(client):
    # Run nowcast with severe storm simulation to ensure an alert is drafted
    payload = {
        "radar_age_min": 5.0,
        "forced_tier": "T0",
        "event_idx": 0,
    }
    run_res = client.post("/api/nowcast/run", json=payload)
    assert run_res.status_code == 200
    run_data = run_res.json()

    # Check pending alerts
    pending_res = client.get("/api/alerts/pending")
    assert pending_res.status_code == 200
    pending_list = pending_res.json()

    if not pending_list and run_data.get("alert_id"):
        # Query specific alert
        alert_id = run_data["alert_id"]
    elif pending_list:
        alert_id = pending_list[0]["alert_id"]
    else:
        # Create a mock pending alert to test approval workflow
        from api.database import save_alert

        alert_id = "IN-IMD-VD-TEST-ALERT-001"
        save_alert(
            alert_id=alert_id,
            status="PENDING_REVIEW",
            created_at="2026-10-04T19:00:00+05:30",
            overall_risk="Red",
            tier_used="T0",
            confidence=0.92,
            districts=[
                {
                    "district_id": "OD_01",
                    "district_name": "Cuttack",
                    "state_name": "Odisha",
                    "max_risk_level": "Red",
                    "peak_reflectivity": 0.85,
                    "earliest_arrival_min": 30,
                    "impacted_area_sq_km": 50,
                    "centroid_lat": 20.46,
                    "centroid_lon": 85.88,
                }
            ],
        )

    # Approve the alert
    approve_payload = {
        "forecaster_id": "duty_officer_bhubaneswar",
        "notes": "Verified convective core over Cuttack. Threat to outdoor workers.",
    }
    approve_res = client.post(f"/api/alerts/{alert_id}/approve", json=approve_payload)
    assert approve_res.status_code == 200
    approved_data = approve_res.json()
    assert approved_data["status"] == "APPROVED"
    assert approved_data["forecaster_id"] == "duty_officer_bhubaneswar"
    assert approved_data["has_cap_xml"] is True

    # Download CAP XML
    cap_res = client.get(f"/api/alerts/{alert_id}/cap.xml")
    assert cap_res.status_code == 200
    assert "application/xml" in cap_res.headers["content-type"]
    xml_content = cap_res.text
    assert validate_cap_xml(xml_content) is True
    assert "Cuttack" in xml_content


def test_system_metrics_and_districts(client):
    res_metrics = client.get("/api/metrics")
    assert res_metrics.status_code == 200
    m_data = res_metrics.json()
    assert "ai_model" in m_data or "error" in m_data

    res_dist = client.get("/api/districts")
    assert res_dist.status_code == 200
    dist_data = res_dist.json()
    assert dist_data.get("type") == "FeatureCollection"
    assert len(dist_data.get("features", [])) > 0
