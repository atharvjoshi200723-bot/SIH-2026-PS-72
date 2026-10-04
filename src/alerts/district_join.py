"""
district_join.py — Spatial intersection of forecasted storm cells with administrative districts.

Maps grid coordinates (128x128 grid at 1 km resolution) to geographical coordinates (lat/lon)
and intersects with district polygons loaded from GeoJSON using Shapely.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from shapely.geometry import Point, shape
from shapely.prepared import prep

from src.config import cfg

logger = logging.getLogger(__name__)


@dataclass
class DistrictImpact:
    """Impact summary for a single district affected by severe weather."""

    district_id: str
    district_name: str
    state_name: str
    max_risk_level: str  # "Green" | "Yellow" | "Orange" | "Red"
    peak_reflectivity: float
    earliest_arrival_min: int  # Lead time in minutes when storm first reaches district
    impacted_area_sq_km: int  # Number of grid cells (1 km² each) above threshold
    centroid_lat: float
    centroid_lon: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DistrictJoinEngine:
    """
    Performs fast spatial intersection between forecast reflectivity fields and district boundaries.
    """

    def __init__(
        self,
        geojson_path: str | Path | None = None,
        config_obj: SimpleNamespace | None = None,
        lat_bounds: tuple[float, float] = (19.7, 20.9),
        lon_bounds: tuple[float, float] = (85.2, 86.4),
    ) -> None:
        """
        Args:
            geojson_path: Path to district GeoJSON file.
            config_obj: Global configuration.
            lat_bounds: (lat_min, lat_max) bounding box of the 128x128 grid.
            lon_bounds: (lon_min, lon_max) bounding box of the 128x128 grid.
        """
        self.cfg = config_obj or cfg
        self.lat_min, self.lat_max = lat_bounds
        self.lon_min, self.lon_max = lon_bounds

        path = Path(geojson_path or self.cfg.paths.district_geojson)
        self.districts = self._load_districts(path)

    def _load_districts(self, path: Path) -> list[dict[str, Any]]:
        """Load GeoJSON features and prepare Shapely polygons for fast querying."""
        if not path.exists():
            raise FileNotFoundError(f"District GeoJSON not found at {path.resolve()}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        loaded = []
        for feature in data.get("features", []):
            geom = shape(feature["geometry"])
            props = feature.get("properties", {})
            loaded.append(
                {
                    "district_id": props.get("district_id", "UNKNOWN"),
                    "district_name": props.get("district_name", "Unknown District"),
                    "state_name": props.get("state_name", "India"),
                    "geom": geom,
                    "prepared": prep(geom),
                    "centroid": geom.centroid,
                }
            )

        logger.info("Loaded %d district geometries from %s", len(loaded), path)
        return loaded

    def grid_to_latlon(
        self, row: int, col: int, H: int = 128, W: int = 128
    ) -> tuple[float, float]:
        """Convert pixel row and column to geographical latitude and longitude."""
        lat = self.lat_max - (row / (H - 1)) * (self.lat_max - self.lat_min)
        lon = self.lon_min + (col / (W - 1)) * (self.lon_max - self.lon_min)
        return float(lat), float(lon)

    def assess_impacts(
        self,
        forecast_grid: np.ndarray,
        threshold: float | None = None,
        step_minutes: int | None = None,
    ) -> list[DistrictImpact]:
        """
        Intersect multi-step forecast grid with district polygons.

        Args:
            forecast_grid: Array of shape (T_out, H, W) in [0, 1].
            threshold: Minimum reflectivity to consider an active hazard (default: yellow_max 0.25).
            step_minutes: Minutes per forecast step (default: 15).

        Returns:
            List of DistrictImpact objects sorted by severity and impact area.
        """
        T_out, H, W = forecast_grid.shape
        step_m = step_minutes or int(self.cfg.time.step_minutes)
        min_thr = threshold if threshold is not None else float(self.cfg.risk.green_max)

        orange_thr = float(self.cfg.risk.yellow_max)
        red_thr = float(self.cfg.risk.orange_max)

        # Pre-compute points for active cells across all time steps
        district_stats: dict[str, dict[str, Any]] = {
            d["district_id"]: {
                "district": d,
                "peak_val": 0.0,
                "earliest_step": None,
                "cell_coords": set(),
            }
            for d in self.districts
        }

        for t in range(T_out):
            frame = forecast_grid[t]
            rows, cols = np.where(frame >= min_thr)

            if len(rows) == 0:
                continue

            for r, c in zip(rows, cols):
                val = float(frame[r, c])
                lat, lon = self.grid_to_latlon(r, c, H, W)
                pt = Point(lon, lat)

                for d in self.districts:
                    d_id = d["district_id"]
                    if d["prepared"].contains(pt):
                        stats = district_stats[d_id]
                        stats["cell_coords"].add((r, c))
                        stats["peak_val"] = max(stats["peak_val"], val)
                        if stats["earliest_step"] is None:
                            stats["earliest_step"] = t
                        break

        impacts: list[DistrictImpact] = []
        for d_id, stats in district_stats.items():
            if len(stats["cell_coords"]) == 0:
                continue

            d = stats["district"]
            peak = stats["peak_val"]
            earliest_step = stats["earliest_step"]
            earliest_min = (
                (earliest_step + 1) * step_m if earliest_step is not None else 0
            )

            if peak < orange_thr:
                risk_lvl = "Yellow"
            elif peak < red_thr:
                risk_lvl = "Orange"
            else:
                risk_lvl = "Red"

            impacts.append(
                DistrictImpact(
                    district_id=d["district_id"],
                    district_name=d["district_name"],
                    state_name=d["state_name"],
                    max_risk_level=risk_lvl,
                    peak_reflectivity=round(peak, 4),
                    earliest_arrival_min=earliest_min,
                    impacted_area_sq_km=len(stats["cell_coords"]),
                    centroid_lat=round(d["centroid"].y, 4),
                    centroid_lon=round(d["centroid"].x, 4),
                )
            )

        # Sort by risk severity (Red > Orange > Yellow) then area descending
        severity_rank = {"Red": 3, "Orange": 2, "Yellow": 1, "Green": 0}
        impacts.sort(
            key=lambda x: (
                severity_rank.get(x.max_risk_level, 0),
                x.impacted_area_sq_km,
            ),
            reverse=True,
        )
        return impacts
