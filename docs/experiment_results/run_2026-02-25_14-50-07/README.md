# Эксперимент run_2026-02-25_14-50-07

Остановлен до завершения 50 эпох; метрики и визуализации получены через backfill по чекпоинту `best.pt`.

**Конфиг:** сплит 90/5/5, full finetune, lr 5e-5, train oversample 4, val/test oversample 2, очищенный датасет.

**Тест (best.pt):** overall mean 2043 m, median 600 m, p90 2549 m; clean n=6845, augmented n=6845; in_overlap n=1660, no_overlap n=12030.

Артефакты: `config.json`, `metrics.json`, `error_histogram.png`, `scatter_errors.png`.
