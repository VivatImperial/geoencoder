#!/usr/bin/env bash
# Обучение на очищенном датасете, 100 эпох, patience 5, Mac GPU (MPS), gradual unfreeze,
# оверсэмплинг, честный val, тест с overlap + метрики по clean/augmented.
set -e
cd "$(dirname "$0")/.."
uv run python geocoding/train.py \
  --csv eda_and_cleaning/addresses_spb_cleaned.csv \
  --epochs 100 \
  --patience 5 \
  --device mps \
  --batch-size 32 \
  --val-batch-size 64 \
  --lr 2e-5 \
  --max-length 256 \
  --head-epochs 2 \
  --unfreeze-every 1 \
  --oversample-factor 2 \
  --test-overlap-n 1000 \
  --seed 42
