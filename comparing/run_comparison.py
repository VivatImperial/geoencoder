#!/usr/bin/env python3
"""
Сравнение нашей модели геокодирования с Dadata и Mistral (LLM по промпту).
Метрики по тестовой выборке в разрезах: аугментированные / неаугментированные.
Оценка стоимости прогона и число пропусков по каждому методу.
Результаты сохраняются в JSON (comparison_report.json) и в Markdown (comparison_summary.md).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

# Загрузка .env при наличии (python-dotenv в dev-зависимостях)
try:
    from dotenv import load_dotenv  # type: ignore[import-untyped]
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math
    R = 6_371_000
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(min(1.0, math.sqrt(a)))
    return R * c


def aggregate_metrics(distances: list[float]) -> dict:
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


def load_predictions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = data.get("predictions", data)
    return data


def geocode_dadata(address: str, api_key: str, secret_key: str) -> tuple[float | None, float | None]:
    """Один запрос к Dadata clean/address. Возвращает (lat, lon) или (None, None) при пропуске."""
    import urllib.request
    url = "https://cleaner.dadata.ru/api/v1/clean/address"
    body = json.dumps([address]).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Token {api_key}",
            "X-Secret": secret_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            out = json.load(resp)
        if not out or not isinstance(out, list):
            return None, None
        item = out[0]
        qc = item.get("qc_geo", 5)
        if qc == 5:  # координаты не определены
            return None, None
        lat_s = item.get("geo_lat")
        lon_s = item.get("geo_lon")
        if lat_s is None or lon_s is None or lat_s == "" or lon_s == "":
            return None, None
        return float(lat_s), float(lon_s)
    except Exception:
        return None, None


def geocode_mistral(address: str, api_key: str) -> tuple[float | None, float | None]:
    """Mistral chat completion: промпт на извлечение lat, lon. Без веб-поиска."""
    import urllib.request
    url = "https://api.mistral.ai/v1/chat/completions"
    prompt = (
        "Ты геокодер. По адресу в России (Санкт-Петербург и область) верни только два числа через пробел: "
        "широта долгота в градусах (WGS84). Пример: 59.934 30.316. Только числа, ничего больше.\nАдрес: "
    ) + address[:500]
    body = json.dumps({
        "model": "mistral-small-latest",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64,
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        return _parse_lat_lon(text)
    except Exception:
        return None, None


def _parse_lat_lon(text: str) -> tuple[float | None, float | None]:
    """Извлекает пару float (lat, lon) из строки."""
    text = text.strip().replace(",", " ")
    nums = re.findall(r"-?\d+\.?\d*", text)
    for i in range(len(nums) - 1):
        try:
            a, b = float(nums[i]), float(nums[i + 1])
            if 50 <= a <= 70 and 19 <= b <= 31:  # примерный bbox СПб
                return a, b
            if 50 <= b <= 70 and 19 <= a <= 31:
                return b, a
        except ValueError:
            continue
    return None, None


def run_comparison(
    predictions_path: Path,
    out_dir: Path,
    sample: int | None = None,
    skip_dadata: bool = False,
    skip_mistral: bool = False,
    rate_limit_dadata: float = 0.06,
) -> None:
    rows = load_predictions(predictions_path)
    if sample is not None and sample < len(rows):
        import random
        random.seed(42)
        rows = random.sample(rows, sample)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Наша модель — уже есть pred в файле
    our_distances = [r["distance_m"] for r in rows]
    our_by_aug = {"clean": [], "augmented": []}
    for r in rows:
        our_by_aug["augmented" if r.get("is_augmented") else "clean"].append(r["distance_m"])

    results = {
        "our_model": {
            "overall": aggregate_metrics(our_distances),
            "clean": aggregate_metrics(our_by_aug["clean"]),
            "augmented": aggregate_metrics(our_by_aug["augmented"]),
            "n_misses": 0,
            "n_total": len(rows),
            "cost_estimate": {"currency": "RUB", "value": 0, "note": "локальная модель"},
        },
    }

    # Dadata
    dadata_key = os.environ.get("DADATA_API_KEY", "").strip()
    dadata_secret = os.environ.get("DADATA_SECRET_KEY", "").strip()
    if not skip_dadata and dadata_key and dadata_secret:
        dists, misses = [], 0
        clean_d, aug_d = [], []
        for i, r in enumerate(rows):
            if rate_limit_dadata and i > 0:
                time.sleep(rate_limit_dadata)
            lat, lon = geocode_dadata(r["address"], dadata_key, dadata_secret)
            if lat is None or lon is None:
                misses += 1
                continue
            d = haversine_m(r["true_lat"], r["true_lon"], lat, lon)
            dists.append(d)
            if r.get("is_augmented"):
                aug_d.append(d)
            else:
                clean_d.append(d)
        results["dadata"] = {
            "overall": aggregate_metrics(dists),
            "clean": aggregate_metrics(clean_d),
            "augmented": aggregate_metrics(aug_d),
            "n_misses": misses,
            "n_total": len(rows),
            "cost_estimate": {
                "currency": "RUB",
                "value": round(len(rows) * 0.20, 2),
                "note": "20 коп/запрос, см. dadata.ru/pricing",
            },
        }
    else:
        results["dadata"] = {"skipped": True, "reason": "no DADATA_API_KEY/DADATA_SECRET_KEY or --skip-dadata"}

    # Mistral
    mistral_key = os.environ.get("MISTRAL_API_KEY", "").strip()
    if not skip_mistral and mistral_key:
        dists, misses = [], 0
        clean_d, aug_d = [], []
        for i, r in enumerate(rows):
            if i > 0:
                time.sleep(0.2)
            lat, lon = geocode_mistral(r["address"], mistral_key)
            if lat is None or lon is None:
                misses += 1
                continue
            d = haversine_m(r["true_lat"], r["true_lon"], lat, lon)
            dists.append(d)
            if r.get("is_augmented"):
                aug_d.append(d)
            else:
                clean_d.append(d)
        # Оценка: ~430 input + 30 output токенов на адрес; $0.05/1M in, $0.08/1M out
        cost_usd = len(rows) * (430 * 0.05 + 30 * 0.08) / 1e6
        results["mistral"] = {
            "overall": aggregate_metrics(dists),
            "clean": aggregate_metrics(clean_d),
            "augmented": aggregate_metrics(aug_d),
            "n_misses": misses,
            "n_total": len(rows),
            "cost_estimate": {
                "currency": "USD",
                "value": round(cost_usd, 4),
                "note": "Mistral Small ~$0.05/1M in, $0.08/1M out",
            },
        }
    else:
        results["mistral"] = {"skipped": True, "reason": "no MISTRAL_API_KEY or --skip-mistral"}

    # Сохранение результатов в JSON и Markdown
    report = {"n_total": len(rows), "predictions_path": str(predictions_path), "methods": results}
    json_path = out_dir / "comparison_report.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Текстовая сводка
    lines = [
        "# Сравнение геокодеров",
        "",
        f"Тестовая выборка: {len(rows)} строк (из {predictions_path}).",
        "",
        "## Метрики по разрезам (clean / augmented)",
        "",
    ]
    for name, data in results.items():
        if data.get("skipped"):
            lines.append(f"### {name}: пропущен — {data.get('reason', '')}")
            lines.append("")
            continue
        lines.append(f"### {name}")
        lines.append(f"- Пропуски: {data['n_misses']} из {data['n_total']} ({100*data['n_misses']/data['n_total']:.1f}%)")
        lines.append(f"- Стоимость (оценка): {data['cost_estimate'].get('value')} {data['cost_estimate'].get('currency')} — {data['cost_estimate'].get('note', '')}")
        for scope in ("overall", "clean", "augmented"):
            m = data.get(scope, {})
            if not m or m.get("n", 0) == 0:
                continue
            lines.append(f"- **{scope}** (n={m['n']}): mean={m['mean_distance_m']:.0f} m, median={m['median_distance_m']:.0f} m, p90={m['p90_distance_m']:.0f} m")
        lines.append("")

    (out_dir / "comparison_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nРезультаты сохранены: JSON — {json_path}, Markdown — {out_dir / 'comparison_summary.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Сравнение нашей модели с Dadata и Mistral")
    parser.add_argument("--predictions", type=str, required=True, help="Путь к test_predictions.json или к каталогу эксперимента")
    parser.add_argument("--out", type=str, default="comparing/results", help="Каталог для отчётов (JSON + MD)")
    parser.add_argument("--sample", type=int, default=None, help="Ограничить число строк для API (для быстрого прогона)")
    parser.add_argument("--skip-dadata", action="store_true", help="Не вызывать Dadata")
    parser.add_argument("--skip-mistral", action="store_true", help="Не вызывать Mistral")
    parser.add_argument("--rate-limit-dadata", type=float, default=0.06, help="Пауза между запросами Dadata (сек), 0 — без паузы")
    args = parser.parse_args()

    path = Path(args.predictions)
    if path.is_dir():
        path = path / "test_predictions.json"
    if not path.is_file():
        raise SystemExit(f"Файл не найден: {path}")

    run_comparison(
        path,
        Path(args.out),
        sample=args.sample,
        skip_dadata=args.skip_dadata,
        skip_mistral=args.skip_mistral,
        rate_limit_dadata=args.rate_limit_dadata,
    )


if __name__ == "__main__":
    main()
