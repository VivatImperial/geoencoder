#!/usr/bin/env python3
"""Обучение BERT-геокодера: адрес -> (lat, lon). Gradual unfreeze, метрики overlap/no_overlap, early stopping."""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
from rich.console import Console
from rich.logging import RichHandler
from torch.utils.data import DataLoader
from tqdm import tqdm

from geocoding.config import BBOX_SPB
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
from geocoding.coordinates import decode_coords
from geocoding.losses import GeocodingMSELoss
from geocoding.metrics import metrics_by_overlap, metrics_by_clean_augmented, haversine_meters
from geocoding.model import GeocodingModel, get_tokenizer

console = Console()


def setup_logging(log_path: Path | None) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        root.addHandler(RichHandler(console=console, show_time=True, rich_tracebacks=True))
    if log_path:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        root.addHandler(fh)


def get_experiment_dir(base: Path) -> Path:
    name = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_config(config: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def _run_metrics(model: GeocodingModel, loader: DataLoader, device: torch.device) -> tuple[list, list, list, list, list, list]:
    """Прогон модели по loader; возврат списков для metrics (in_train, is_augmented)."""
    model.eval()
    all_lat_norm, all_lon_norm = [], []
    all_lat, all_lon = [], []
    all_in_train = []
    all_is_augmented = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pred = model(input_ids=input_ids, attention_mask=attention_mask)
            all_lat_norm.extend(pred[:, 0].cpu().tolist())
            all_lon_norm.extend(pred[:, 1].cpu().tolist())
            all_lat.extend(batch["lat"])
            all_lon.extend(batch["lon"])
            all_in_train.extend(batch["in_train"])
            all_is_augmented.extend(batch.get("is_augmented", [False] * len(batch["lat"])))
    return all_lat_norm, all_lon_norm, all_lat, all_lon, all_in_train, all_is_augmented


def _compute_val_loss(
    model: GeocodingModel,
    val_loader: DataLoader,
    device: torch.device,
    criterion: GeocodingMSELoss,
) -> float | None:
    """Средний MSE (нормализованные координаты) на валидации."""
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            lat_norm = batch["lat_norm"].to(device)
            lon_norm = batch["lon_norm"].to(device)
            pred = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(pred, lat_norm, lon_norm)
            if torch.isfinite(loss):
                total_loss += loss.item() * input_ids.size(0)
                n += input_ids.size(0)
    return total_loss / n if n else None


def compute_val_metrics(model: GeocodingModel, val_loader: DataLoader, device: torch.device) -> dict:
    """Метрики валидации: overall, clean, augmented (отдельно по чистым и аугментированным)."""
    t = _run_metrics(model, val_loader, device)
    *overlap_args, is_aug = t
    by_ca = metrics_by_clean_augmented(*overlap_args[:4], is_aug)
    return {
        "overall": by_ca["overall"],
        "n_overall": by_ca["n_overall"],
        "clean": by_ca["clean"],
        "augmented": by_ca["augmented"],
        "n_clean": by_ca["n_clean"],
        "n_augmented": by_ca["n_augmented"],
    }


def collect_test_predictions(
    model: GeocodingModel,
    test_loader: DataLoader,
    device: torch.device,
    address_max_len: int = 100,
) -> list[dict]:
    """Собирает по каждому тестовому сэмплу: true_lat, true_lon, pred_lat, pred_lon, distance_m, address, is_augmented, in_train."""
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pred = model(input_ids=input_ids, attention_mask=attention_mask)
            addresses = batch.get("address", [""] * len(batch["lat"]))
            for i in range(len(batch["lat"])):
                tl, tlon = batch["lat"][i], batch["lon"][i]
                pn_lat, pn_lon = pred[i, 0].item(), pred[i, 1].item()
                pl, plon = decode_coords(pn_lat, pn_lon)
                d = haversine_meters(tl, tlon, pl, plon)
                addr = (addresses[i] if i < len(addresses) else "")[:address_max_len]
                rows.append({
                    "true_lat": tl,
                    "true_lon": tlon,
                    "pred_lat": pl,
                    "pred_lon": plon,
                    "distance_m": round(d, 2),
                    "address": addr,
                    "is_augmented": batch["is_augmented"][i] if i < len(batch["is_augmented"]) else False,
                    "in_train": batch["in_train"][i] if i < len(batch["in_train"]) else False,
                })
    return rows


def compute_test_metrics(model: GeocodingModel, test_loader: DataLoader, device: torch.device) -> dict:
    """Метрики теста: overall, in_overlap, no_overlap + clean, augmented."""
    t = _run_metrics(model, test_loader, device)
    *overlap_args, is_aug = t
    by_overlap = metrics_by_overlap(*overlap_args)
    by_ca = metrics_by_clean_augmented(overlap_args[0], overlap_args[1], overlap_args[2], overlap_args[3], is_aug)
    return {
        **by_overlap,
        "clean": by_ca["clean"],
        "augmented": by_ca["augmented"],
        "n_clean": by_ca["n_clean"],
        "n_augmented": by_ca["n_augmented"],
    }


def plot_history(history: list[dict], out_dir: Path) -> None:
    """Графики обучения по эпохам: loss, метрики val (mean/median/p90/p95), доля NaN-батчей."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss_raw = [h.get("val_loss") for h in history]
    val_loss = [v if v is not None else float("nan") for v in val_loss_raw]
    vm = [h["val_metrics"] for h in history]
    mean_overall = [m["overall"]["mean_distance_m"] for m in vm]
    mean_clean = [m.get("clean", {}).get("mean_distance_m") for m in vm]
    mean_aug = [m.get("augmented", {}).get("mean_distance_m") for m in vm]
    median_overall = [m["overall"].get("median_distance_m") for m in vm]
    p90_overall = [m["overall"].get("p90_distance_m") for m in vm]
    p95_overall = [m["overall"].get("p95_distance_m") for m in vm]
    nan_ratio = [h.get("nan_ratio", 0) * 100 for h in history]
    nan_batches = [h.get("nan_batches", 0) for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Loss (train + val loss)
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, label="train loss", color="C0")
    if any(v is not None for v in val_loss_raw):
        ax.plot(epochs, val_loss, label="val loss", color="C1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.set_title("Loss")
    ax.grid(True, alpha=0.3)

    # Val mean distance (overall / clean / augmented)
    ax = axes[0, 1]
    ax.plot(epochs, mean_overall, label="val mean overall (m)", color="C0")
    if any(v is not None for v in mean_clean):
        ax.plot(epochs, mean_clean, label="val mean clean (m)", color="C1")
    if any(v is not None for v in mean_aug):
        ax.plot(epochs, mean_aug, label="val mean augmented (m)", color="C2")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean distance (m)")
    ax.legend()
    ax.set_title("Validation: mean error")
    ax.grid(True, alpha=0.3)

    # Val overall: median, p90, p95
    ax = axes[1, 0]
    if any(v is not None for v in median_overall):
        ax.plot(epochs, median_overall, label="median (m)", color="C0")
    if any(v is not None for v in p90_overall):
        ax.plot(epochs, p90_overall, label="p90 (m)", color="C1")
    if any(v is not None for v in p95_overall):
        ax.plot(epochs, p95_overall, label="p95 (m)", color="C2")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Distance (m)")
    ax.legend()
    ax.set_title("Validation: overall distribution")
    ax.grid(True, alpha=0.3)

    # NaN batches / ratio
    ax = axes[1, 1]
    ax2 = ax.twinx()
    (l1,) = ax.plot(epochs, nan_batches, label="NaN batches", color="C0")
    (l2,) = ax2.plot(epochs, nan_ratio, label="NaN %", color="C1", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NaN batches", color="C0")
    ax2.set_ylabel("NaN %", color="C1")
    ax.legend(handles=[l1, l2], loc="upper right")
    ax.set_title("Training stability")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "history.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train geocoding BERT (address -> lat, lon)")
    parser.add_argument("--csv", type=str, default="data/addresses_spb.csv", help="Path to addresses CSV")
    parser.add_argument("--experiments-dir", type=str, default="geocoding/experiments", help="Base dir for experiment outputs")
    parser.add_argument("--model-name", type=str, default="deepvk/USER2-small", help="HuggingFace encoder name")
    parser.add_argument("--batch-size", type=int, default=32, help="Train batch size")
    parser.add_argument("--val-batch-size", type=int, default=64, help="Validation batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Peak learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--max-length", type=int, default=256, help="Max token length")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience (epochs)")
    parser.add_argument("--unfreeze-every", type=int, default=1, help="Unfreeze one encoder layer every N epochs")
    parser.add_argument("--head-epochs", type=int, default=2, help="Epochs with only head trained before unfreezing")
    parser.add_argument("--encoder-lr-mult", type=float, default=0.2, help="LR multiplier for newly unfrozen encoder layers (stability)")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--head-only", action="store_true", help="Train only the head (no gradual unfreeze); stable baseline")
    parser.add_argument("--full-finetune", action="store_true", help="Unfreeze all encoder layers from epoch 1 (no gradual unfreeze)")
    parser.add_argument("--nan-stop-ratio", type=float, default=0.05, help="Stop and restore best if NaN batch ratio exceeds this")
    parser.add_argument("--no-augment", action="store_true", help="Disable augmentations")
    parser.add_argument("--oversample-factor", type=int, default=2, metavar="N", help="Oversampling: add N-1 augmented copies per sample for val/test (default 2)")
    parser.add_argument("--train-oversample-factor", type=int, default=None, metavar="N", help="Oversampling для train (по умолчанию = oversample-factor; можно задать больше, напр. 4)")
    parser.add_argument("--no-sage", action="store_true", help="Disable SAGE typos")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda, mps, cpu or auto (default)")
    parser.add_argument("--resume", type=str, default=None, metavar="PATH", help="Resume training: load model weights from checkpoint (.pt) and train for --epochs more")
    parser.add_argument("--test-overlap-n", type=int, default=1000, help="Add N samples from train to test so test has in_overlap (default 1000)")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Доля train (default 0.9)")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Доля val (default 0.05)")
    parser.add_argument("--test-ratio", type=float, default=0.05, help="Доля test (default 0.05)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    def _resolve_device() -> torch.device:
        if args.device and args.device.lower() != "auto":
            return torch.device(args.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        try:
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                return torch.device("mps")  # Apple Silicon (Metal)
        except Exception:
            pass
        return torch.device("cpu")

    train_oversample = args.train_oversample_factor if args.train_oversample_factor is not None else args.oversample_factor
    exp_dir = get_experiment_dir(Path(args.experiments_dir))
    setup_logging(exp_dir / "train_log.txt")

    config = {
        "csv": args.csv,
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "max_length": args.max_length,
        "patience": args.patience,
        "unfreeze_every": args.unfreeze_every,
        "head_epochs": args.head_epochs,
        "encoder_lr_mult": args.encoder_lr_mult,
        "grad_clip": args.grad_clip,
        "head_only": args.head_only,
        "full_finetune": args.full_finetune,
        "nan_stop_ratio": args.nan_stop_ratio,
        "val_honest": True,
        "resume": args.resume,
        "test_overlap_n": args.test_overlap_n,
        "bbox": list(BBOX_SPB),
        "augment": not args.no_augment,
        "oversample_factor": args.oversample_factor,
        "train_oversample_factor": train_oversample if not args.no_augment else None,
        "sage_typos": not args.no_sage,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
    }
    device = _resolve_device()
    config["device"] = str(device)
    save_config(config, exp_dir / "config.json")
    logging.info("Device: %s", device)

    rows = load_csv(args.csv)
    logging.info("Loaded %d rows from %s", len(rows), args.csv)
    train_data, val_data, test_data = train_val_test_split(
        rows,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.seed,
    )
    train_norm_set = build_train_normalized_set(train_data)
    val_size_before = len(val_data)
    val_data = [r for r in val_data if normalize_address(r["address"]) not in train_norm_set]
    test_size_before = len(test_data)
    if args.test_overlap_n > 0 and len(train_data) > 0:
        rng = random.Random(args.seed)
        n_add = min(args.test_overlap_n, len(train_data))
        test_data = test_data + rng.sample(train_data, n_add)
        logging.info("Test: added %d samples from train for overlap; test size %d -> %d", n_add, test_size_before, len(test_data))
    logging.info(
        "Train %d, val %d (honest: only unseen addresses, filtered from %d), test %d; train normalized set size %d",
        len(train_data), len(val_data), val_size_before, len(test_data), len(train_norm_set),
    )
    if len(val_data) == 0:
        logging.warning("Val is empty after filtering (all val addresses appear in train). Early stopping and best checkpoint will not use val.")

    tokenizer = get_tokenizer(args.model_name)
    augmentor = None
    if not args.no_augment:
        augmentor = AddressAugmentor(
            shuffle_prob=0.35,
            junk_add_prob=0.25,
            junk_remove_prob=0.3,
            typo_prob=0.5,
            typo_use_sage=not args.no_sage,
            seed=args.seed,
        )

    if augmentor is not None and (train_oversample > 1 or args.oversample_factor > 1):
        train_data = oversample_with_augment(train_data, augmentor, train_oversample, seed=args.seed)
        val_data = oversample_with_augment(val_data, augmentor, args.oversample_factor, seed=args.seed + 1)
        test_data = oversample_with_augment(test_data, augmentor, args.oversample_factor, seed=args.seed + 2)
        logging.info(
            "Oversampling train=%d val/test=%d: train=%d val=%d test=%d (with augmented copies)",
            train_oversample, args.oversample_factor, len(train_data), len(val_data), len(test_data),
        )
    augmentor = None  # аугментации уже в данных (оверсэмплинг), в датасете не применяем

    train_ds = GeocodingDataset(train_data, train_normalized_set=train_norm_set, is_train=True, augmentor=augmentor)
    val_ds = GeocodingDataset(val_data, train_normalized_set=train_norm_set, is_train=False, augmentor=augmentor)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_geocoding(b, tokenizer, args.max_length),
        num_workers=0,
        pin_memory=(device.type == "cuda"),  # только для CUDA
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_geocoding(b, tokenizer, args.max_length),
        num_workers=0,
    )
    test_ds = GeocodingDataset(test_data, train_normalized_set=train_norm_set, is_train=False, augmentor=augmentor)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_geocoding(b, tokenizer, args.max_length),
        num_workers=0,
    )

    model = GeocodingModel(encoder_name=args.model_name, hidden_size=384, dropout=0.1)
    model.to(device)
    if not args.full_finetune:
        model.freeze_encoder()
    else:
        logging.info("Full finetune: all encoder layers trainable from epoch 1")
    if args.resume:
        path = Path(args.resume)
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=device, weights_only=True)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=True)
        logging.info("Resumed model weights from %s (training %d epochs)", path, args.epochs)
    num_layers = model.num_encoder_layers()
    if num_layers == 0:
        num_layers = 12
    logging.info("Encoder layers: %d", num_layers)

    criterion = GeocodingMSELoss(reduction="mean")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.01, (total_steps - step) / (total_steps - warmup_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = []
    best_mean_m = float("inf")
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    unfrozen_layers: set[int] = set()

    for epoch in range(args.epochs):
        # Gradual unfreeze (top-down): after head_epochs, unfreeze one layer every unfreeze_every epochs. Отключено при --head-only и --full-finetune.
        if not args.head_only and not args.full_finetune and epoch >= args.head_epochs and num_layers > 0:
            k = epoch - args.head_epochs
            layer_idx = num_layers - 1 - (k // args.unfreeze_every)
            if layer_idx >= 0 and layer_idx not in unfrozen_layers:
                new_params = model.unfreeze_encoder_layer(layer_idx)
                if new_params:
                    unfrozen_layers.add(layer_idx)
                    encoder_lr = args.lr * args.encoder_lr_mult
                    optimizer.add_param_group({"params": new_params, "lr": encoder_lr})
                    if hasattr(scheduler, "base_lrs"):
                        scheduler.base_lrs.append(encoder_lr)
                    if hasattr(scheduler, "lr_lambdas"):
                        scheduler.lr_lambdas.append(scheduler.lr_lambdas[0])
                    logging.info("Unfroze encoder layer %d (lr=%.2e)", layer_idx, encoder_lr)

        model.train()
        train_loss_sum = 0.0
        train_n = 0
        nan_batches = 0
        total_batches = len(train_loader)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            lat_norm = batch["lat_norm"].to(device)
            lon_norm = batch["lon_norm"].to(device)
            optimizer.zero_grad()
            pred = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(pred, lat_norm, lon_norm)
            if not torch.isfinite(loss):
                nan_batches += 1
                if nan_batches <= 3 or nan_batches % 100 == 0:
                    logging.warning("Skipping batch with non-finite loss (nan/inf), total skipped this epoch: %d", nan_batches)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            train_loss_sum += loss.item() * input_ids.size(0)
            train_n += input_ids.size(0)
            pbar.set_postfix(loss=loss.item())
        train_loss = train_loss_sum / train_n if train_n else float("nan")
        nan_ratio = nan_batches / total_batches if total_batches else 0.0

        val_metrics = compute_val_metrics(model, val_loader, device)
        mean_m = val_metrics["overall"]["mean_distance_m"]
        val_loss = _compute_val_loss(model, val_loader, device, criterion)

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "nan_batches": nan_batches,
            "nan_ratio": nan_ratio,
            "val_loss": val_loss,
            "val_metrics": {
                "overall": val_metrics["overall"],
                "n_overall": val_metrics["n_overall"],
                "clean": val_metrics.get("clean"),
                "augmented": val_metrics.get("augmented"),
                "n_clean": val_metrics.get("n_clean"),
                "n_augmented": val_metrics.get("n_augmented"),
            },
        }
        history.append(record)

        log_extra = ""
        if val_metrics.get("n_clean") is not None and val_metrics.get("n_augmented") is not None:
            log_extra = " clean=%.2f aug=%.2f" % (
                val_metrics["clean"]["mean_distance_m"],
                val_metrics["augmented"]["mean_distance_m"],
            )
        val_loss_str = (" val_loss=%.6f" % val_loss) if val_loss is not None else ""
        logging.info(
            "Epoch %d train_loss=%.6f%s nan_batches=%d (%.1f%%) val mean_distance_m=%.2f%s",
            epoch + 1, train_loss, val_loss_str, nan_batches, nan_ratio * 100, mean_m, log_extra,
        )

        if val_metrics["n_overall"] > 0:
            if mean_m < best_mean_m:
                best_mean_m = mean_m
            # Early stopping по val_loss (если есть), иначе по mean_distance_m
            use_loss = val_loss is not None
            if use_loss:
                improved = val_loss < best_val_loss
                if improved:
                    best_val_loss = val_loss
            else:
                improved = mean_m < best_mean_m
            if improved:
                best_epoch = epoch + 1
                patience_counter = 0
                torch.save({"model": model.state_dict(), "epoch": epoch + 1}, exp_dir / "best.pt")
            else:
                patience_counter += 1
            if patience_counter >= args.patience:
                logging.info("Early stopping at epoch %d (by val_loss=%s)", epoch + 1, "yes" if use_loss else "val mean_m")
                break
        if nan_ratio >= args.nan_stop_ratio and (exp_dir / "best.pt").exists():
            logging.warning(
                "Too many NaN batches (%.1f%%). Restoring best checkpoint and stopping.",
                nan_ratio * 100,
            )
            ckpt = torch.load(exp_dir / "best.pt", map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model"])
            break

    torch.save({"model": model.state_dict(), "epoch": epoch + 1}, exp_dir / "last.pt")
    if best_epoch == -1:
        torch.save({"model": model.state_dict(), "epoch": epoch + 1}, exp_dir / "best.pt")
        logging.info("Val was empty; saved last model as best.pt")

    test_metrics = None
    if (exp_dir / "best.pt").exists() and len(test_data) > 0:
        ckpt = torch.load(exp_dir / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        test_metrics = compute_test_metrics(model, test_loader, device)
        test_predictions = collect_test_predictions(model, test_loader, device)
        with (exp_dir / "test_predictions.json").open("w", encoding="utf-8") as f:
            json.dump(test_predictions, f, ensure_ascii=False, indent=2)
        logging.info("Saved %d test predictions to %s", len(test_predictions), exp_dir / "test_predictions.json")
        logging.info(
            "Test (best): overall=%.2f m | in_overlap=%.2f m (n=%d) no_overlap=%.2f m (n=%d) | clean=%.2f m (n=%d) augmented=%.2f m (n=%d)",
            test_metrics["overall"]["mean_distance_m"],
            test_metrics["in_overlap"]["mean_distance_m"], test_metrics["n_in_overlap"],
            test_metrics["no_overlap"]["mean_distance_m"], test_metrics["n_no_overlap"],
            test_metrics.get("clean", {}).get("mean_distance_m", 0), test_metrics.get("n_clean", 0),
            test_metrics.get("augmented", {}).get("mean_distance_m", 0), test_metrics.get("n_augmented", 0),
        )

    metrics_out = {"best_epoch": best_epoch, "best_mean_distance_m": best_mean_m, "history": history}
    if test_metrics is not None:
        metrics_out["test_metrics"] = test_metrics
    with (exp_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False)

    try:
        plot_history(history, exp_dir)
    except Exception as e:
        logging.warning("Could not save plots: %s", e)

    if (exp_dir / "test_predictions.json").exists():
        try:
            from geocoding.visualize_test import _load_predictions, build_histogram, build_map_html, build_scatter
            random.seed(args.seed)
            rows = _load_predictions(exp_dir / "test_predictions.json")
            build_map_html(rows, exp_dir / "map_errors.html", add_segments=True)
            build_histogram(rows, exp_dir / "error_histogram.png")
            build_scatter(rows, exp_dir / "scatter_errors.png", sample=5000)
            logging.info("Test visualizations saved: map_errors.html, error_histogram.png, scatter_errors.png")
        except Exception as e:
            logging.warning("Could not build test visualizations: %s", e)

    logging.info("Done. Best epoch %d, best mean_distance_m %.2f. Results in %s", best_epoch, best_mean_m, exp_dir)


if __name__ == "__main__":
    main()
