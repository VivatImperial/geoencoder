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


def _popup_html(label: str, address: str, distance_m: float, is_true: bool) -> str:
    """HTML для popup: подпись (Истина/Предсказание), адрес, ошибка в метрах."""
    addr_esc = address.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"<div style='min-width:180px;'><b>{label}</b><br/>{addr_esc}<br/><small>Ошибка: {distance_m:.0f} м</small></div>"


def _map_interaction_js() -> str:
    """JS: подсветка линии при клике на точку/линию, перелёт по двойному клику между истиной и прогнозом."""
    # Не используем буквально </script> в строке, чтобы не закрыть тег при вставке в HTML
    return r"""
(function(){
  function findMap() {
    for (var k in window) {
      try {
        if (window[k] && window[k]._leaflet_id != null && typeof window[k].getCenter === 'function')
          return window[k];
      } catch (e) {}
    }
    return null;
  }
  function latLngEq(a, b) {
    if (!a || !b) return false;
    return Math.abs(a.lat - b.lat) < 1e-5 && Math.abs(a.lng - b.lng) < 1e-5;
  }
  var map = findMap();
  if (!map) return;
  var markers = [], polylines = [];
  map.eachLayer(function(layer) {
    if (layer.getLatLng) markers.push(layer);
    else if (layer.getLatLngs && layer.getLatLngs().length === 2) polylines.push(layer);
  });
  var pairs = [];
  polylines.forEach(function(line) {
    var pts = line.getLatLngs();
    var m1 = null, m2 = null;
    markers.forEach(function(m) {
      var L = m.getLatLng();
      if (latLngEq(L, pts[0])) m1 = m;
    });
    markers.forEach(function(m) {
      if (m === m1) return;
      var L = m.getLatLng();
      if (latLngEq(L, pts[1])) { m2 = m; return; }
    });
    if (m1 && m2) pairs.push({ trueMarker: m1, predMarker: m2, line: line });
  });
  var defaultStyles = {};
  pairs.forEach(function(p) {
    defaultStyles[p.line._leaflet_id] = {
      weight: p.line.options.weight,
      opacity: p.line.options.opacity,
      color: p.line.options.color
    };
    p.trueRadius = p.trueMarker.options.radius != null ? p.trueMarker.options.radius : 4;
    p.predRadius = p.predMarker.options.radius != null ? p.predMarker.options.radius : 3;
  });
  function unhighlightAll() {
    pairs.forEach(function(p) {
      var opt = defaultStyles[p.line._leaflet_id];
      if (opt) p.line.setStyle({ weight: opt.weight, opacity: opt.opacity, color: opt.color });
      if (p.trueMarker.setRadius) {
        p.trueMarker.setRadius(p.trueRadius);
        p.predMarker.setRadius(p.predRadius);
      }
    });
  }
  function highlight(pair) {
    unhighlightAll();
    pair.line.setStyle({ weight: 6, opacity: 1, color: '#e74c3c' });
    if (pair.trueMarker.setRadius) {
      pair.trueMarker.setRadius(6);
      pair.predMarker.setRadius(5);
    }
  }
  function bindPair(pair) {
    function onSingleClick(e) { if (e) e.originalEvent.stopPropagation(); highlight(pair); }
    function onDblClickTrue(e) { if (e) e.originalEvent.stopPropagation(); map.flyTo(pair.predMarker.getLatLng(), Math.max(map.getZoom(), 16)); }
    function onDblClickPred(e) { if (e) e.originalEvent.stopPropagation(); map.flyTo(pair.trueMarker.getLatLng(), Math.max(map.getZoom(), 16)); }
    pair.trueMarker.on({ click: onSingleClick, dblclick: onDblClickTrue });
    pair.predMarker.on({ click: onSingleClick, dblclick: onDblClickPred });
    pair.line.on({ click: onSingleClick });
  }
  pairs.forEach(bindPair);
  map.on('click', function() { unhighlightAll(); });
})();
"""


def build_map_html(
    rows: list[dict],
    out_path: Path,
    center_lat: float = 59.93,
    center_lon: float = 30.31,
    zoom: int = 11,
    max_points: int | None = None,
    add_segments: bool = True,
    show_correct: bool = True,
    correct_threshold_m: float = 100.0,
) -> None:
    try:
        import folium
        from branca.element import Element
    except ImportError:
        raise SystemExit("Установите folium: uv add folium") from None

    if max_points is not None and len(rows) > max_points:
        rows = random.sample(rows, max_points)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles="OpenStreetMap")

    if show_correct:
        correct_rows = [r for r in rows if r["distance_m"] < correct_threshold_m]
        error_rows = [r for r in rows if r["distance_m"] >= correct_threshold_m]
        fg_correct = folium.FeatureGroup(name="Правильные (<100 м)")
        fg_errors = folium.FeatureGroup(name="Ошибки")
        for r in correct_rows:
            lat, lon = r["true_lat"], r["true_lon"]
            d = r["distance_m"]
            addr = r.get("address", "")[:100]
            tip = f"Истина: {addr} — {d:.0f} м"
            popup_html = _popup_html("Истина", addr, d, is_true=True)
            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                color="green",
                fill=True,
                fillColor="green",
                fillOpacity=0.8,
                tooltip=tip,
                popup=folium.Popup(popup_html, max_width=250),
            ).add_to(fg_correct)
            folium.PolyLine(
                locations=[[r["true_lat"], r["true_lon"]], [r["pred_lat"], r["pred_lon"]]],
                color="green",
                weight=1,
                opacity=0.5,
                dash_array="2, 4",
                popup=f"Правильно — {d:.0f} м",
            ).add_to(fg_correct)
            folium.CircleMarker(
                location=[r["pred_lat"], r["pred_lon"]],
                radius=2,
                color="darkgreen",
                fill=True,
                fillColor="darkgreen",
                fillOpacity=0.9,
                tooltip=f"Предсказание — {d:.0f} м (двойной клик → к истине)",
                popup=folium.Popup(_popup_html("Предсказание", addr, d, is_true=False), max_width=250),
            ).add_to(fg_correct)
        for r in error_rows:
            lat, lon = r["true_lat"], r["true_lon"]
            d = r["distance_m"]
            addr = r.get("address", "")[:100]
            tip = f"Истина: {addr} — ошибка {d:.0f} м"
            popup_html = _popup_html("Истина", addr, d, is_true=True)
            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                color=_color_by_error(d),
                fill=True,
                fillColor=_color_by_error(d),
                fillOpacity=0.7,
                tooltip=tip,
                popup=folium.Popup(popup_html, max_width=250),
            ).add_to(fg_errors)
        if add_segments:
            for r in error_rows:
                d = r["distance_m"]
                addr = r.get("address", "")[:100]
                folium.PolyLine(
                    locations=[[r["true_lat"], r["true_lon"]], [r["pred_lat"], r["pred_lon"]]],
                    color="blue",
                    weight=2,
                    opacity=0.6,
                    dash_array="8, 8",
                    popup=f"Ошибка {r['distance_m']:.0f} м (клик — подсветка, двойной клик по точке — перелёт)",
                ).add_to(fg_errors)
                folium.CircleMarker(
                    location=[r["pred_lat"], r["pred_lon"]],
                    radius=3,
                    color="gray",
                    fill=True,
                    fillColor="gray",
                    fillOpacity=0.8,
                    tooltip=f"Предсказание — {d:.0f} м (двойной клик → к истине)",
                    popup=folium.Popup(_popup_html("Предсказание", addr, d, is_true=False), max_width=250),
                ).add_to(fg_errors)
        fg_correct.add_to(m)
        fg_errors.add_to(m)
    else:
        for r in rows:
            lat, lon = r["true_lat"], r["true_lon"]
            d = r["distance_m"]
            addr = r.get("address", "")[:100]
            tip = f"{addr} — ошибка {d:.0f} м"
            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                color=_color_by_error(d),
                fill=True,
                fillColor=_color_by_error(d),
                fillOpacity=0.7,
                tooltip=tip,
                popup=folium.Popup(_popup_html("Истина", addr, d, is_true=True), max_width=250),
            ).add_to(m)
        if add_segments and len(rows) >= 1:
            for r in rows:
                d = r["distance_m"]
                addr = r.get("address", "")[:100]
                folium.PolyLine(
                    locations=[[r["true_lat"], r["true_lon"]], [r["pred_lat"], r["pred_lon"]]],
                    color="blue",
                    weight=2,
                    opacity=0.6,
                    dash_array="8, 8",
                    popup=f"Ошибка {r['distance_m']:.0f} м",
                ).add_to(m)
                folium.CircleMarker(
                    location=[r["pred_lat"], r["pred_lon"]],
                    radius=3,
                    color="gray",
                    fill=True,
                    fillColor="gray",
                    fillOpacity=0.8,
                    tooltip=f"Предсказание — {d:.0f} м",
                    popup=folium.Popup(_popup_html("Предсказание", addr, d, is_true=False), max_width=250),
                ).add_to(m)

    folium.LayerControl().add_to(m)
    if add_segments and len(rows) > 0:
        m.get_root().script.add_child(Element(_map_interaction_js()))
    m.save(str(out_path))


def build_comparison_map_html(
    map_data_path: Path,
    out_path: Path,
    center_lat: float = 59.93,
    center_lon: float = 30.31,
    zoom: int = 11,
    max_points: int = 500,
) -> None:
    """Карта сравнения: истинная точка + прогнозы нашей модели, Dadata и Mistral."""
    try:
        import folium
    except ImportError:
        raise SystemExit("Установите folium: uv add folium") from None

    with map_data_path.open(encoding="utf-8") as f:
        rows = json.load(f)
    if not rows:
        return
    if len(rows) > max_points:
        rows = random.sample(rows, max_points)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles="OpenStreetMap")

    fg_true = folium.FeatureGroup(name="Истинная точка")
    fg_our = folium.FeatureGroup(name="Наша модель")
    fg_dadata = folium.FeatureGroup(name="Dadata")
    fg_mistral = folium.FeatureGroup(name="Mistral")
    fg_segments = folium.FeatureGroup(name="Отрезки истинная → прогноз")

    for r in rows:
        tlat, tlon = r["true_lat"], r["true_lon"]
        addr = r.get("address", "")[:80]

        folium.CircleMarker(
            location=[tlat, tlon],
            radius=5,
            color="green",
            fill=True,
            fillColor="green",
            fillOpacity=0.9,
            tooltip=f"Истина: {addr}",
            popup=f"Истина: {addr}",
        ).add_to(fg_true)

        folium.CircleMarker(
            location=[r["our_lat"], r["our_lon"]],
            radius=4,
            color="blue",
            fill=True,
            fillColor="blue",
            fillOpacity=0.8,
            tooltip=f"Наша модель: {r['our_distance_m']:.0f} м",
            popup=f"Наша модель: {addr} — ошибка {r['our_distance_m']:.0f} м",
        ).add_to(fg_our)
        folium.PolyLine(
            [[tlat, tlon], [r["our_lat"], r["our_lon"]]],
            color="blue",
            weight=1,
            opacity=0.5,
            dash_array="4, 4",
        ).add_to(fg_segments)

        if r.get("dadata_lat") is not None and r.get("dadata_lon") is not None:
            folium.CircleMarker(
                location=[r["dadata_lat"], r["dadata_lon"]],
                radius=4,
                color="orange",
                fill=True,
                fillColor="orange",
                fillOpacity=0.8,
                tooltip="Dadata",
                popup=f"Dadata: {addr}",
            ).add_to(fg_dadata)
            folium.PolyLine(
                [[tlat, tlon], [r["dadata_lat"], r["dadata_lon"]]],
                color="orange",
                weight=1,
                opacity=0.5,
                dash_array="4, 4",
            ).add_to(fg_segments)

        if r.get("mistral_lat") is not None and r.get("mistral_lon") is not None:
            folium.CircleMarker(
                location=[r["mistral_lat"], r["mistral_lon"]],
                radius=4,
                color="purple",
                fill=True,
                fillColor="purple",
                fillOpacity=0.8,
                tooltip="Mistral",
                popup=f"Mistral: {addr}",
            ).add_to(fg_mistral)
            folium.PolyLine(
                [[tlat, tlon], [r["mistral_lat"], r["mistral_lon"]]],
                color="purple",
                weight=1,
                opacity=0.5,
                dash_array="4, 4",
            ).add_to(fg_segments)

    for fg in (fg_true, fg_our, fg_dadata, fg_mistral, fg_segments):
        fg.add_to(m)
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
