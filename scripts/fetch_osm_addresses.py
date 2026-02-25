#!/usr/bin/env python3
"""
Сбор пар «адрес – координаты» из локального PBF-дампера OSM.
Скачиваешь актуальный срез региона (BBBike/Geofabrik), запускаешь — получаешь CSV без лимитов и API.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import osmium
from rich.console import Console
from rich.logging import RichHandler
from tqdm import tqdm

# Bbox Санкт-Петербурга (юг, запад, север, восток)
BBOX_SPB = (59.70, 29.60, 60.15, 30.85)

# По умолчанию — Северо-Западный ФО (СПб входит), Geofabrik, стабильная раздача
DEFAULT_PBF_URL = "https://download.geofabrik.de/russia/northwestern-fed-district-latest.osm.pbf"

LOG_INTERVAL_OBJ = 50_000
LOG_INTERVAL_SEC = 30

# Порядок полей для сборки строки адреса (от крупного к мелкому)
ADDR_KEYS = (
    "addr:country",
    "addr:state",
    "addr:province",
    "addr:region",
    "addr:city",
    "addr:district",
    "addr:subdistrict",
    "addr:county",
    "addr:suburb",
    "addr:hamlet",
    "addr:street",
    "addr:place",
    "addr:block",
    "addr:block_number",
    "addr:housenumber",
    "addr:housename",
    "addr:unit",
    "addr:flats",
    "addr:postcode",
    "addr:postbox",
)
# Все addr:* ключи, по которым пускаем объект в выборку (максимально широко)
ADDR_FILTER_KEYS = list(ADDR_KEYS) + [
    "addr:full",
    "addr:door",
    "addr:floor",
    "addr:plot",
]


def setup_logging(console: Console) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)


def build_address(tags: dict) -> str:
    """Собирает строку адреса: addr:full приоритетен, иначе по порядку ADDR_KEYS, иначе все addr:* подряд."""
    full = tags.get("addr:full")
    if full and isinstance(full, str) and full.strip():
        return full.strip()
    parts = [tags.get(k) for k in ADDR_KEYS if tags.get(k)]
    s = ", ".join(p.strip() for p in parts if isinstance(p, str) and p.strip())
    if s:
        return s
    # Любой addr:* тег, чтобы не терять объекты с редкими ключами
    fallback = [v.strip() for k, v in tags.items() if k.startswith("addr:") and v and isinstance(v, str) and v.strip()]
    return ", ".join(fallback) if fallback else ""


def normalize_address(s: str) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def looks_like_number_range_only(addr: str) -> bool:
    """Отсекаем мусор: адрес из диапазонов чисел (addr:flats / housenumber range) без улицы/города.
    Любая длина: и "447-525", и длинные "4-13;23-32;42-51;…" считаем мусором."""
    if not addr:
        return False
    s = addr.strip()
    # Нет букв (слова) и при этом только цифры/дефисы/;,. пробелы — мусор
    if not re.search(r"[а-яёa-z]{2,}", s, re.IGNORECASE):
        if re.match(r"^[\d\s\-;,.]+$", s):
            return True
    return False


def tags_has_addr(tags: osmium.TagList) -> bool:
    for t in tags:
        if t.k.startswith("addr:") and t.v and t.v.strip():
            return True
    return False


def tags_to_dict(tags: osmium.TagList) -> dict:
    return {t.k: t.v for t in tags}


def way_center(nodes: list) -> tuple[float, float] | None:
    lats, lons = [], []
    for n in nodes:
        if n.location.valid():
            lats.append(n.lat)
            lons.append(n.lon)
    if not lats:
        return None
    return (sum(lats) / len(lats), sum(lons) / len(lons))


def in_bbox(lat: float, lon: float, bbox: tuple[float, float, float, float]) -> bool:
    south, west, north, east = bbox
    return south <= lat <= north and west <= lon <= east


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Извлечь адреса из OSM PBF (локальный дамп) в CSV"
    )
    parser.add_argument(
        "pbf",
        nargs="?",
        default=os.environ.get("OSM_PBF_PATH", ""),
        help="Путь к .osm.pbf или URL (по умолчанию — скачать BBBike СПб)",
    )
    parser.add_argument(
        "--bbox",
        default=os.environ.get("BBOX_SPB", ""),
        help="south,west,north,east (по умолчанию СПб)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=os.environ.get("OUTPUT_PATH", "./data/addresses_spb.csv"),
        help="Выходной CSV",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Дополнительно записать JSON",
    )
    args = parser.parse_args()

    console = Console()
    setup_logging(console)
    console.print("[bold green]Сбор адресов из OSM PBF (локальный дамп)[/bold green]")

    if args.bbox:
        parts = [p.strip() for p in args.bbox.split(",")]
        bbox = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])) if len(parts) == 4 else BBOX_SPB
    else:
        bbox = BBOX_SPB

    pbf_path = args.pbf.strip() or DEFAULT_PBF_URL
    delete_pbf = False
    if pbf_path.startswith("http://") or pbf_path.startswith("https://"):
        pbf_url = pbf_path
        console.print("[yellow]Скачиваю PBF...[/yellow]")
        fd, pbf_path = tempfile.mkstemp(suffix=".osm.pbf")
        os.close(fd)
        delete_pbf = True
        urllib.request.urlretrieve(pbf_url, pbf_path)
        console.print("[green]PBF загружен.[/green]")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Промежуточная запись: после каждой новой уникальной записи дописываем в CSV
    seen: set[str] = set()
    unique_list: list[dict] = []
    out_file = open(out_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(out_file)
    csv_writer.writerow(["address", "lat", "lon"])
    out_file.flush()

    key_filter = osmium.filter.KeyFilter(*ADDR_FILTER_KEYS)
    fp = (
        osmium.FileProcessor(pbf_path, osmium.osm.NODE | osmium.osm.WAY)
        .with_filter(key_filter)
        .with_locations()
    )

    total_read = 0
    last_log_time = time.monotonic()
    for obj in tqdm(fp, desc="Чтение PBF", unit="obj", file=sys.stderr, dynamic_ncols=False, mininterval=1.0):
        total_read += 1
        if not tags_has_addr(obj.tags):
            continue
        tags = tags_to_dict(obj.tags)
        addr = build_address(tags)
        if not addr:
            continue
        if obj.is_node():
            lat, lon = obj.lat, obj.lon
        elif obj.is_way():
            center = way_center(list(obj.nodes))
            if not center:
                continue
            lat, lon = center
        else:
            continue
        if not in_bbox(lat, lon, bbox):
            continue

        if looks_like_number_range_only(addr):
            continue
        key = normalize_address(addr)
        if not key or key in seen:
            continue
        seen.add(key)
        unique_list.append({"address": addr, "lat": lat, "lon": lon})
        csv_writer.writerow([addr, lat, lon])
        out_file.flush()

        now = time.monotonic()
        if total_read % LOG_INTERVAL_OBJ == 0 or (now - last_log_time) >= LOG_INTERVAL_SEC:
            logging.info(
                "Обработано объектов %s, уникальных записей в файле %s",
                total_read, len(unique_list),
            )
            last_log_time = now

    if delete_pbf:
        try:
            os.unlink(pbf_path)
        except OSError:
            pass

    out_file.close()

    console.print(f"[cyan]Прочитано объектов: {total_read}[/cyan]")
    console.print(f"[cyan]Уникальных адресов записано: [green]{len(unique_list)}[/green][/cyan]")
    console.print(f"[bold green]Сохранено: [link=file://{out_path}]{out_path}[/link][/bold green]")

    if args.json:
        json_path = out_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(unique_list, f, ensure_ascii=False, indent=2)
        console.print(f"[bold green]JSON: {json_path}[/bold green]")

    console.print("[bold green]Готово.[/bold green]")


if __name__ == "__main__":
    main()
