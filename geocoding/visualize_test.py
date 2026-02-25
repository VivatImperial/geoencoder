#!/usr/bin/env python3
"""
Визуализация тестовых предсказаний: карта ошибок, гистограмма, scatter.
Требует: test_predictions.json (сохраняется в experiment dir после обучения).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _color_by_error(m: float) -> str:
    if m < 100:
        return "green"
    if m < 500:
        return "yellow"
    if m < 1000:
        return "orange"
    return "red"


def _load_predictions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def build_map_html(
    rows: list[dict],
    out_path: Path,
    center_lat: float = 59.93,
    center_lon: float = 30.31,
    zoom: int = 11,
    max_points: int | None = None,
    add_segments: bool = True,
) -> None:
    try:
        import folium
    except ImportError:
        raise SystemExit("Установите folium: uv add folium") from None

    if max_points is not None and len(rows) > max_points:
        rows = random.sample(rows, max_points)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles="OpenStreetMap")

    for r in rows:
        lat, lon = r["true_lat"], r["true_lon"]
        d = r["distance_m"]
        addr = r.get("address", "")[:80]
        tip = f"{addr} — ошибка {d:.0f} м"
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color=_color_by_error(d),
            fill=True,
            fillColor=_color_by_error(d),
            fillOpacity=0.7,
            tooltip=tip,
            popup=tip,
        ).add_to(m)

    if add_segments and len(rows) >= 1:
        for r in rows:
            # Пунктирная линия от правильной точки (true) до предсказания (pred)
            folium.PolyLine(
                locations=[[r["true_lat"], r["true_lon"]], [r["pred_lat"], r["pred_lon"]]],
                color="blue",
                weight=2,
                opacity=0.6,
                dash_array="8, 8",
                popup=f"Ошибка {r['distance_m']:.0f} м",
            ).add_to(m)
            # Точка предсказания (конец отрезка ошибки)
            folium.CircleMarker(
                location=[r["pred_lat"], r["pred_lon"]],
                radius=3,
                color="gray",
                fill=True,
                fillColor="gray",
                fillOpacity=0.8,
                tooltip=f"Предсказание — ошибка {r['distance_m']:.0f} м",
                popup=f"Предсказание — ошибка {r['distance_m']:.0f} м",
            ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(str(out_path))


def build_histogram(rows: list[dict], out_path: Path, bins: int = 80, xmax: float | None = 2000) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dists = [r["distance_m"] for r in rows]
    plt.figure(figsize=(10, 5))
    plt.hist(dists, bins=bins, color="steelblue", edgecolor="white", alpha=0.8)
    if xmax is not None:
        plt.xlim(0, xmax)
    plt.xlabel("Ошибка (м)")
    plt.ylabel("Число семплов")
    plt.title(f"Распределение ошибок на тесте (n={len(dists)})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def build_scatter(rows: list[dict], out_path: Path, sample: int = 5000) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(rows) > sample:
        rows = random.sample(rows, sample)
    true_lat = [r["true_lat"] for r in rows]
    true_lon = [r["true_lon"] for r in rows]
    pred_lat = [r["pred_lat"] for r in rows]
    pred_lon = [r["pred_lon"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(true_lat, pred_lat, alpha=0.3, s=5)
    axes[0].plot([59.7, 60.15], [59.7, 60.15], "r--", lw=1, label="y=x")
    axes[0].set_xlabel("Истинная широта")
    axes[0].set_ylabel("Предсказанная широта")
    axes[0].set_title("Широта")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(true_lon, pred_lon, alpha=0.3, s=5)
    axes[1].plot([29.6, 30.85], [29.6, 30.85], "r--", lw=1, label="y=x")
    axes[1].set_xlabel("Истинная долгота")
    axes[1].set_ylabel("Предсказанная долгота")
    axes[1].set_title("Долгота")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.suptitle(f"Истинные vs предсказанные координаты (n={len(rows)})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Визуализация тестовых предсказаний геокодера")
    parser.add_argument("predictions", type=str, nargs="?", default=None, help="Путь к test_predictions.json или к каталогу эксперимента")
    parser.add_argument("--out-dir", type=str, default=None, help="Каталог для сохранения карты и графиков (по умолчанию — рядом с JSON)")
    parser.add_argument("--map-points", type=int, default=None, metavar="N", help="Макс. точек на карте (по умолчанию — все)")
    parser.add_argument("--no-segments", action="store_true", help="Не рисовать отрезки истинная→предсказанная на карте")
    parser.add_argument("--no-map", action="store_true", help="Не строить интерактивную карту")
    parser.add_argument("--no-hist", action="store_true", help="Не строить гистограмму")
    parser.add_argument("--no-scatter", action="store_true", help="Не строить scatter")
    parser.add_argument("--seed", type=int, default=42, help="Seed для сэмплирования")
    args = parser.parse_args()

    random.seed(args.seed)

    path = Path(args.predictions or "")
    if not path.exists():
        parser.error("Укажите путь к test_predictions.json или к каталогу эксперимента (где лежит test_predictions.json)")
    if path.is_dir():
        path = path / "test_predictions.json"
    if not path.is_file():
        parser.error(f"Файл не найден: {path}")

    rows = _load_predictions(path)
    if not rows:
        raise SystemExit("Нет записей в test_predictions.json")

    out_dir = Path(args.out_dir) if args.out_dir else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_map:
        build_map_html(
            rows,
            out_dir / "map_errors.html",
            max_points=args.map_points,
            add_segments=not args.no_segments,
        )
        print(f"Карта: {out_dir / 'map_errors.html'}")
    if not args.no_hist:
        build_histogram(rows, out_dir / "error_histogram.png")
        print(f"Гистограмма: {out_dir / 'error_histogram.png'}")
    if not args.no_scatter:
        build_scatter(rows, out_dir / "scatter_errors.png")
        print(f"Scatter: {out_dir / 'scatter_errors.png'}")
    print("Готово.")


if __name__ == "__main__":
    main()
