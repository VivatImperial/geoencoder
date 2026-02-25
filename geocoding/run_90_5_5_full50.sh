#!/usr/bin/env bash
# Train 90% / val 5% / test 5%, full finetune с эпохи 1, 50 эпох, повышенный LR, агрессивные аугментации, val loss.
# Очищенные данные (bbox + дедупликация в load_csv и EDA).
set -e
cd "$(dirname "$0")/.."
uv run python geocoding/train.py \
  --csv eda_and_cleaning/addresses_spb_cleaned.csv \
  --train-ratio 0.9 \
  --val-ratio 0.05 \
  --test-ratio 0.05 \
  --epochs 50 \
  --patience 5 \
  --device mps \
  --batch-size 32 \
  --val-batch-size 64 \
  --lr 5e-5 \
  --max-length 256 \
  --full-finetune \
  --oversample-factor 2 \
  --train-oversample-factor 4 \
  --test-overlap-n 1000 \
  --seed 42
