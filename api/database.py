"""
database.py — Lightweight SQLite persistence for alerts, forecast runs, and forecaster audit logs.

No external daemon required (no Redis, no Celery, no Postgres).
Thread-safe SQLite connection with automatic schema migration.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from src.config import cfg

_lock = threading.Lock()


def get_db_path() -> Path:
    """Resolve database path from configuration."""
    db_path = Path(cfg.api.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def init_db(db_path: Path | None = None) -> None:
    """Initialise SQLite tables for alerts, runs, and audit logs."""
    path = db_path or get_db_path()
    with _lock, sqlite3.connect(path) as conn:
        cursor = conn.cursor()

        # Alerts table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                overall_risk TEXT NOT NULL,
                tier_used TEXT NOT NULL,
                confidence REAL NOT NULL,
                districts_json TEXT NOT NULL,
                forecaster_id TEXT,
                forecaster_notes TEXT,
                reviewed_at TEXT,
                rejection_reason TEXT,
                cap_xml TEXT
            )
            """
        )

        # Forecast runs table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS forecast_runs (
                run_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                tier_used TEXT NOT NULL,
                confidence REAL NOT NULL,
                latency_ms REAL NOT NULL,
                max_risk TEXT NOT NULL,
                sensor_status_json TEXT NOT NULL,
                risk_summary_json TEXT NOT NULL
            )
            """
        )

        # Audit logs table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,
                actor TEXT NOT NULL,
                details TEXT
            )
            """
        )
        conn.commit()


def save_forecast_run(
    run_id: str,
    timestamp: str,
    tier_used: str,
    confidence: float,
    latency_ms: float,
    max_risk: str,
    sensor_status: dict[str, Any],
    risk_summary: list[dict[str, Any]],
    db_path: Path | None = None,
) -> None:
    path = db_path or get_db_path()
    with _lock, sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO forecast_runs 
            (run_id, timestamp, tier_used, confidence, latency_ms, max_risk, sensor_status_json, risk_summary_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                timestamp,
                tier_used,
                confidence,
                latency_ms,
                max_risk,
                json.dumps(sensor_status),
                json.dumps(risk_summary),
            ),
        )
        conn.commit()


def save_alert(
    alert_id: str,
    status: str,
    created_at: str,
    overall_risk: str,
    tier_used: str,
    confidence: float,
    districts: list[dict[str, Any]],
    db_path: Path | None = None,
) -> None:
    path = db_path or get_db_path()
    with _lock, sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO alerts 
            (alert_id, status, created_at, overall_risk, tier_used, confidence, districts_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                alert_id,
                status,
                created_at,
                overall_risk,
                tier_used,
                confidence,
                json.dumps(districts),
            ),
        )
        conn.commit()


def update_alert(
    alert_id: str,
    status: str,
    forecaster_id: str,
    reviewed_at: str,
    notes: str | None = None,
    rejection_reason: str | None = None,
    cap_xml: str | None = None,
    db_path: Path | None = None,
) -> None:
    path = db_path or get_db_path()
    with _lock, sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE alerts 
            SET status = ?, forecaster_id = ?, forecaster_notes = ?, reviewed_at = ?,
                rejection_reason = ?, cap_xml = ?
            WHERE alert_id = ?
            """,
            (
                status,
                forecaster_id,
                notes,
                reviewed_at,
                rejection_reason,
                cap_xml,
                alert_id,
            ),
        )
        conn.commit()


def get_alert(alert_id: str, db_path: Path | None = None) -> dict[str, Any] | None:
    path = db_path or get_db_path()
    with _lock, sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM alerts WHERE alert_id = ?", (alert_id,))
        row = cursor.fetchone()
        if not row:
            return None
        d = dict(row)
        d["districts"] = json.loads(d["districts_json"])
        return d


def list_alerts(
    status: str | None = None, db_path: Path | None = None
) -> list[dict[str, Any]]:
    path = db_path or get_db_path()
    with _lock, sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if status:
            cursor.execute(
                "SELECT * FROM alerts WHERE status = ? ORDER BY created_at DESC",
                (status,),
            )
        else:
            cursor.execute("SELECT * FROM alerts ORDER BY created_at DESC")
        rows = cursor.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["districts"] = json.loads(d["districts_json"])
            result.append(d)
        return result


def log_audit(
    action: str, actor: str, details: str, timestamp: str, db_path: Path | None = None
) -> None:
    path = db_path or get_db_path()
    with _lock, sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO audit_logs (timestamp, action, actor, details) VALUES (?, ?, ?, ?)",
            (timestamp, action, actor, details),
        )
        conn.commit()


def get_audit_logs(
    limit: int = 50, db_path: Path | None = None
) -> list[dict[str, Any]]:
    path = db_path or get_db_path()
    with _lock, sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM audit_logs ORDER BY log_id DESC LIMIT ?", (limit,)
        )
        return [dict(r) for r in cursor.fetchall()]
