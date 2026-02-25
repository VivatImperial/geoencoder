#!/usr/bin/env python3
"""
EDA и очистка датасета адресов.
Загружает CSV, применяет правила из geocoding.cleaning, пишет отчёт и очищенный датасет.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from geocoding.cleaning import filter_rows
from geocoding.config import BBOX_SPB


def load_csv_raw(path: Path) -> list[dict]:
    """Загрузить все строки CSV без фильтрации."""
    rows = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            addr = (row.get("address") or "").strip()
            try:
                lat = float(row.get("lat", 0))
                lon = float(row.get("lon", 0))
            except (ValueError, TypeError):
                continue
            rows.append({"address": addr, "lat": lat, "lon": lon})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA и очистка датасета адресов")
    parser.add_argument("--csv", type=str, default="data/addresses_spb.csv", help="Входной CSV")
    parser.add_argument("--out-dir", type=str, default="eda_and_cleaning", help="Каталог для отчёта и очищенного CSV")
    parser.add_argument("--cleaned-csv", type=str, default="", help="Имя файла очищенного CSV (по умолчанию addresses_spb_cleaned.csv в out-dir)")
    args = parser.parse_args()

    src = Path(args.csv)
    if not src.is_file():
        raise FileNotFoundError(f"CSV не найден: {src}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_csv_raw(src)
    kept, dropped = filter_rows(rows)

    by_reason: dict[str, list[dict]] = defaultdict(list)
    for r in dropped:
        by_reason[r["_drop_reason"]].append(r)

    # Bbox и дедупликация координат (как в geocoding.data.load_csv)
    lat_min, lon_min, lat_max, lon_max = BBOX_SPB
    in_bbox = [r for r in kept if lat_min <= r["lat"] <= lat_max and lon_min <= r["lon"] <= lon_max]
    out_bbox_count = len(kept) - len(in_bbox)
    seen = set()
    final = []
    for r in in_bbox:
        key = (round(r["lat"], 6), round(r["lon"], 6))
        if key in seen:
            continue
        seen.add(key)
        final.append(r)
    dup_dropped = len(in_bbox) - len(final)

    summary = {
        "source_csv": str(src),
        "total_rows": len(rows),
        "kept_after_cleaning": len(kept),
        "dropped_cleaning": len(dropped),
        "drop_ratio": round(len(dropped) / len(rows), 4) if rows else 0,
        "out_bbox": out_bbox_count,
        "after_bbox": len(in_bbox),
        "duplicate_coords_dropped": dup_dropped,
        "final_rows": len(final),
        "by_reason": {k: len(v) for k, v in sorted(by_reason.items())},
        "examples_per_reason": {
            k: [x["address"][:80] for x in v[:15]]
            for k, v in sorted(by_reason.items())
        },
    }

    report_lines = [
        "# EDA и очистка датасета адресов",
        "",
        f"**Источник:** `{src}`",
        f"**Всего строк (сырых):** {summary['total_rows']}",
        f"**После очистки адресов:** {summary['kept_after_cleaning']} (отсеяно {summary['dropped_cleaning']}, {summary['drop_ratio']*100:.2f}%)",
        f"**Вне bbox (СПб):** отсеяно {summary['out_bbox']}, осталось {summary['after_bbox']}",
        f"**Дубликаты (lat,lon):** отсеяно {summary['duplicate_coords_dropped']}",
        f"**Итого (финальный датасет):** {summary['final_rows']}",
        "",
        "## Причины отсева (адреса)",
        "",
    ]
    for reason, count in sorted(summary["by_reason"].items(), key=lambda x: -x[1]):
        report_lines.append(f"- **{reason}**: {count}")
        for ex in summary["examples_per_reason"].get(reason, [])[:8]:
            report_lines.append(f"  - `{ex}`")
        report_lines.append("")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Отчёт: {report_path}")

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Сводка: {summary_path}")

    cleaned_name = args.cleaned_csv or "addresses_spb_cleaned.csv"
    cleaned_path = out_dir / cleaned_name
    with cleaned_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["address", "lat", "lon"])
        w.writeheader()
        for r in final:
            w.writerow({"address": r["address"], "lat": r["lat"], "lon": r["lon"]})
    print(f"Очищенный CSV: {cleaned_path} ({len(final)} строк)")

    dropped_path = out_dir / "dropped_examples.json"
    dropped_export = []
    for reason, items in sorted(by_reason.items()):
        for r in items[:30]:
            dropped_export.append({"address": r["address"], "reason": reason, "lat": r["lat"], "lon": r["lon"]})
    with dropped_path.open("w", encoding="utf-8") as f:
        json.dump(dropped_export, f, ensure_ascii=False, indent=2)
    print(f"Примеры отсеянных: {dropped_path}")


if __name__ == "__main__":
    main()
