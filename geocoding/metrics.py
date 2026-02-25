"""Метрики в метрах: Haversine, агрегация по overlap/no_overlap."""
from __future__ import annotations

import math
from typing import Any

from geocoding.coordinates import decode_coords

# Радиус Земли в метрах
EARTH_RADIUS_M = 6_371_000


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Расстояние по поверхности сферы между двумя точками в метрах."""
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(min(1.0, a)))
    return EARTH_RADIUS_M * c


def distances_meters_batch(
    pred_lat_norm: list[float],
    pred_lon_norm: list[float],
    true_lat: list[float],
    true_lon: list[float],
) -> list[float]:
    """Список расстояний в метрах для батча (после decode pred в градусы)."""
    out = []
    for i in range(len(true_lat)):
        lat_dec, lon_dec = decode_coords(pred_lat_norm[i], pred_lon_norm[i])
        d = haversine_meters(lat_dec, lon_dec, true_lat[i], true_lon[i])
        out.append(d)
    return out


def aggregate_metrics(distances: list[float]) -> dict[str, float | int]:
    """Среднее, медиана, p90, p95 и число семплов n по списку расстояний в метрах."""
    if not distances:
        return {"mean_distance_m": 0.0, "median_distance_m": 0.0, "p90_distance_m": 0.0, "p95_distance_m": 0.0, "n": 0}
    s = sorted(distances)
    n = len(s)
    return {
        "mean_distance_m": sum(s) / n,
        "median_distance_m": s[n // 2],
        "p90_distance_m": s[int(0.9 * n)] if n else 0.0,
        "p95_distance_m": s[int(0.95 * n)] if n else 0.0,
        "n": n,
    }


def metrics_by_overlap(
    pred_lat_norm: list[float],
    pred_lon_norm: list[float],
    true_lat: list[float],
    true_lon: list[float],
    in_train: list[bool],
) -> dict[str, Any]:
    """Считает метрики overall, in_overlap (in_train), no_overlap (not in_train)."""
    dists = distances_meters_batch(pred_lat_norm, pred_lon_norm, true_lat, true_lon)
    overall = aggregate_metrics(dists)
    in_overlap_d = [d for d, it in zip(dists, in_train) if it]
    no_overlap_d = [d for d, it in zip(dists, in_train) if not it]
    in_overlap = aggregate_metrics(in_overlap_d)
    no_overlap = aggregate_metrics(no_overlap_d)
    return {
        "overall": overall,
        "in_overlap": in_overlap,
        "no_overlap": no_overlap,
        "n_overall": len(dists),
        "n_in_overlap": len(in_overlap_d),
        "n_no_overlap": len(no_overlap_d),
    }


def metrics_by_clean_augmented(
    pred_lat_norm: list[float],
    pred_lon_norm: list[float],
    true_lat: list[float],
    true_lon: list[float],
    is_augmented: list[bool],
) -> dict[str, Any]:
    """Считает метрики overall, clean (не аугментированные), augmented (аугментированные)."""
    dists = distances_meters_batch(pred_lat_norm, pred_lon_norm, true_lat, true_lon)
    overall = aggregate_metrics(dists)
    clean_d = [d for d, aug in zip(dists, is_augmented) if not aug]
    aug_d = [d for d, aug in zip(dists, is_augmented) if aug]
    return {
        "overall": overall,
        "clean": aggregate_metrics(clean_d),
        "augmented": aggregate_metrics(aug_d),
        "n_overall": len(dists),
        "n_clean": len(clean_d),
        "n_augmented": len(aug_d),
    }
