"""
Дополняет эксперимент: тестовые предсказания, визуализации, обновлённый history.png.
Запуск: uv run python geocoding/backfill_experiment.py [путь к эксперименту]
По умолчанию берётся последний каталог в geocoding/experiments/ (по имени).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from geocoding.data import (
    build_train_normalized_set,
    collate_geocoding,
    load_csv,
    normalize_address,
    oversample_with_augment,
    train_val_test_split,
    GeocodingDataset,
)
from geocoding.augmentations import AddressAugmentor
from geocoding.model import GeocodingModel, get_tokenizer


def _resolve_device(device_str: str | None) -> torch.device:
    if device_str and device_str.lower() not in ("auto", ""):
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Дополнить эксперимент: test_predictions, визуализации, history.png")
    parser.add_argument("experiment_dir", type=str, nargs="?", default=None, help="Каталог эксперимента (по умолчанию — последний run_*)")
    parser.add_argument("--device", type=str, default=None, help="cuda, mps, cpu (по умолчанию из config)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    import os
    os.chdir(repo_root)

    experiments_dir = Path("geocoding/experiments")
    if args.experiment_dir:
        exp_dir = Path(args.experiment_dir)
        if not exp_dir.is_absolute():
            exp_dir = experiments_dir / exp_dir if (experiments_dir / args.experiment_dir).exists() else exp_dir
    else:
        runs = sorted(experiments_dir.glob("run_*"), key=lambda p: p.name)
        if not runs:
            raise SystemExit("Нет каталогов run_* в geocoding/experiments/")
        exp_dir = runs[-1]
    if not exp_dir.is_dir():
        raise SystemExit(f"Каталог не найден: {exp_dir}")

    config_path = exp_dir / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"Нет config.json в {exp_dir}")
    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)

    best_path = exp_dir / "best.pt"
    if not best_path.is_file():
        raise SystemExit(f"Нет best.pt в {exp_dir}")

    csv_path = Path(config.get("csv", "eda_and_cleaning/addresses_spb_cleaned.csv"))
    if not csv_path.is_file():
        raise SystemExit(f"CSV не найден: {csv_path} (запускайте из корня репозитория)")

    seed = config.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)

    device = _resolve_device(args.device or config.get("device"))
    model_name = config.get("model_name", "deepvk/USER2-small")
    batch_size = config.get("batch_size", 32)
    val_batch_size = config.get("val_batch_size", 64)
    max_length = config.get("max_length", 256)
    test_overlap_n = config.get("test_overlap_n", 1000)
    oversample_factor = config.get("oversample_factor", 2)
    use_augment = config.get("augment", True)
    use_sage = config.get("sage_typos", True)

    rows = load_csv(csv_path)
    train_data, val_data, test_data = train_val_test_split(rows, random_state=seed)
    train_norm_set = build_train_normalized_set(train_data)
    val_data = [r for r in val_data if normalize_address(r["address"]) not in train_norm_set]
    if test_overlap_n > 0 and len(train_data) > 0:
        n_add = min(test_overlap_n, len(train_data))
        test_data = test_data + random.sample(train_data, n_add)

    tokenizer = get_tokenizer(model_name)
    augmentor = None
    if use_augment:
        augmentor = AddressAugmentor(typo_use_sage=use_sage, seed=seed)
    if augmentor is not None and oversample_factor > 1:
        train_data = oversample_with_augment(train_data, augmentor, oversample_factor, seed=seed)
        val_data = oversample_with_augment(val_data, augmentor, oversample_factor, seed=seed + 1)
        test_data = oversample_with_augment(test_data, augmentor, oversample_factor, seed=seed + 2)
    augmentor = None

    test_ds = GeocodingDataset(test_data, train_normalized_set=train_norm_set, is_train=False, augmentor=augmentor)
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_ds,
        batch_size=val_batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_geocoding(b, tokenizer, max_length),
        num_workers=0,
    )

    model = GeocodingModel(encoder_name=model_name, hidden_size=384, dropout=0.1)
    ckpt = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    from geocoding.train import collect_test_predictions, compute_test_metrics, plot_history

    test_predictions = collect_test_predictions(model, test_loader, device)
    with (exp_dir / "test_predictions.json").open("w", encoding="utf-8") as f:
        json.dump(test_predictions, f, ensure_ascii=False, indent=2)
    print(f"Saved test_predictions.json ({len(test_predictions)} rows)")

    test_metrics = compute_test_metrics(model, test_loader, device)
    metrics_path = exp_dir / "metrics.json"
    if metrics_path.is_file():
        with metrics_path.open(encoding="utf-8") as f:
            metrics_out = json.load(f)
    else:
        metrics_out = {"best_epoch": None, "best_mean_distance_m": test_metrics["overall"]["mean_distance_m"], "history": []}
    metrics_out["test_metrics"] = test_metrics
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False)
    print("Updated metrics.json with test_metrics (clean, augmented, n)")

    try:
        from geocoding.visualize_test import _load_predictions, build_histogram, build_map_html, build_scatter
        random.seed(seed)
        rows_viz = _load_predictions(exp_dir / "test_predictions.json")
        build_map_html(rows_viz, exp_dir / "map_errors.html", add_segments=True)
        build_histogram(rows_viz, exp_dir / "error_histogram.png")
        build_scatter(rows_viz, exp_dir / "scatter_errors.png", sample=5000)
        print("Saved map_errors.html, error_histogram.png, scatter_errors.png")
    except Exception as e:
        print(f"Visualizations failed: {e}")

    if "history" in metrics_out and metrics_out["history"]:
        try:
            plot_history(metrics_out["history"], exp_dir)
            print("Regenerated history.png (2x2: loss, val mean, val distribution, NaN)")
        except Exception as e:
            print(f"plot_history failed: {e}")

    print("Done.", exp_dir)


if __name__ == "__main__":
    main()
