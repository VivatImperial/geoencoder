"""Кодирование/декодирование координат в [0, 1] по bbox для стабильного обучения."""
from __future__ import annotations

from geocoding.config import BBOX_SPB, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN


def encode_coords(lat: float, lon: float) -> tuple[float, float]:
    """Преобразует градусы в нормализованные значения [0, 1]."""
    lat_norm = (lat - LAT_MIN) / (LAT_MAX - LAT_MIN)
    lon_norm = (lon - LON_MIN) / (LON_MAX - LON_MIN)
    return (lat_norm, lon_norm)


def decode_coords(lat_norm: float, lon_norm: float) -> tuple[float, float]:
    """Обратное преобразование из [0, 1] в градусы."""
    lat = lat_norm * (LAT_MAX - LAT_MIN) + LAT_MIN
    lon = lon_norm * (LON_MAX - LON_MIN) + LON_MIN
    return (lat, lon)


def get_bbox() -> tuple[float, float, float, float]:
    """Возвращает (lat_min, lon_min, lat_max, lon_max)."""
    return BBOX_SPB
