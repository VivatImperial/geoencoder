"""Загрузка CSV, сплит train/val/test, overlap, Dataset и collate с динамическим паддингом."""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from geocoding.augmentations import AddressAugmentor
from geocoding.coordinates import encode_coords
from geocoding.cleaning import is_bad_address
from geocoding.config import BBOX_SPB


def oversample_with_augment(
    data: list[dict[str, Any]],
    augmentor: AddressAugmentor,
    factor: int,
    seed: int = 0,
    address_key: str = "address",
) -> list[dict[str, Any]]:
    """Оверсэмплинг: к каждому объекту добавляем (factor - 1) копий с аугментированным адресом. factor=1 — без изменений."""
    if factor <= 1:
        return data
    out: list[dict[str, Any]] = []
    for i, row in enumerate(data):
        out.append({**dict(row), "_is_augmented": False})
        for k in range(factor - 1):
            aug_addr = augmentor(row[address_key], seed=seed + i * 10000 + k)
            out.append({**row, address_key: aug_addr, "_is_augmented": True})
    return out


def normalize_address(s: str) -> str:
    """Нормализация адреса для сравнения (lowercase, strip, collapse spaces)."""
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def load_csv(
    path: str | Path,
    bbox: tuple[float, float, float, float] | None = BBOX_SPB,
    drop_duplicate_coords: bool = True,
) -> list[dict[str, Any]]:
    """Читает CSV с колонками address, lat, lon. Отсеивает неполные/странные адреса (cleaning), точки вне bbox, при необходимости — дубликаты (lat,lon)."""
    path = Path(path)
    lat_min, lon_min, lat_max, lon_max = bbox if bbox else (-90, -180, 90, 180)
    rows = []
    seen_coords: set[tuple[float, float]] = set()
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            addr = (row.get("address") or "").strip()
            try:
                lat = float(row.get("lat", 0))
                lon = float(row.get("lon", 0))
            except (ValueError, TypeError):
                continue
            if not addr or is_bad_address(addr):
                continue
            if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
                continue
            if drop_duplicate_coords:
                key = (round(lat, 6), round(lon, 6))
                if key in seen_coords:
                    continue
                seen_coords.add(key)
            rows.append({"address": addr, "lat": lat, "lon": lon})
    return rows


def train_val_test_split(
    rows: list[dict[str, Any]],
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    random_state: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Разбивает данные на train/val/test. random_state для воспроизводимости."""
    import random

    rng = random.Random(random_state)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    n = len(indices)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val = n - n_train
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    train_data = [rows[i] for i in train_idx]
    val_data = [rows[i] for i in val_idx]
    test_data = [rows[i] for i in test_idx]
    return train_data, val_data, test_data


def build_train_normalized_set(train_data: list[dict[str, Any]]) -> set[str]:
    """Множество нормализованных адресов из train для определения overlap."""
    return {normalize_address(r["address"]) for r in train_data}


class GeocodingDataset(Dataset):
    """Датасет: адрес -> (lat_norm, lon_norm). При train=True применяются аугментации."""

    def __init__(
        self,
        data: list[dict[str, Any]],
        train_normalized_set: set[str] | None = None,
        is_train: bool = False,
        augmentor: AddressAugmentor | None = None,
    ):
        self.data = data
        self.train_normalized_set = train_normalized_set or set()
        self.is_train = is_train
        self.augmentor = augmentor

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.data[idx]
        address = row["address"]
        lat, lon = row["lat"], row["lon"]
        lat_norm, lon_norm = encode_coords(lat, lon)

        if self.is_train and self.augmentor is not None:
            address = self.augmentor(address, seed=idx)

        in_train = normalize_address(address) in self.train_normalized_set
        is_augmented = row.get("_is_augmented", False)

        return {
            "address": address,
            "lat_norm": lat_norm,
            "lon_norm": lon_norm,
            "lat": lat,
            "lon": lon,
            "in_train": in_train,
            "is_augmented": is_augmented,
        }


def collate_geocoding(
    batch: list[dict],
    tokenizer: Any,
    max_length: int = 512,
) -> dict[str, torch.Tensor | list]:
    """Собирает батч: динамический паддинг до макс. длины в батче (но не больше max_length)."""
    addresses = [b["address"] for b in batch]
    enc = tokenizer(
        addresses,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True,
    )
    lat_norm = torch.tensor([b["lat_norm"] for b in batch], dtype=torch.float32)
    lon_norm = torch.tensor([b["lon_norm"] for b in batch], dtype=torch.float32)
    in_train = [b["in_train"] for b in batch]
    is_augmented = [b.get("is_augmented", False) for b in batch]
    lat = [b["lat"] for b in batch]
    lon = [b["lon"] for b in batch]
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "lat_norm": lat_norm,
        "lon_norm": lon_norm,
        "in_train": in_train,
        "is_augmented": is_augmented,
        "lat": lat,
        "lon": lon,
        "address": addresses,
    }
