# Эксперимент run_2026-02-25_14-50-07

Остановлен до завершения 50 эпох; метрики и визуализации получены через backfill по чекпоинту `best.pt` (без переобучения).

## Конфигурация

| Параметр | Значение |
|----------|----------|
| Датасет | `eda_and_cleaning/addresses_spb_cleaned.csv` |
| Сплит | train 90% / val 5% / test 5% |
| Модель | deepvk/USER2-small, full finetune |
| Learning rate | 5e-5, warmup 10% |
| Обучающая выборка | oversample ×4 (с аугментацией) |
| Val/Test oversample | ×2 |
| Аугментация | включена (SAGE-опечатки) |
| test_overlap_n | 1000 (добавление примеров из train в test для оценки overlap) |
| bbox | [59.7, 29.6, 60.15, 30.85] (Санкт-Петербург и окрестности) |
| seed | 42 |

## Метрики на тесте (best.pt)

- **Объём:** n = 13 690 (clean: 6845, augmented: 6845).
- **Overlap:** in_overlap n=1660, no_overlap n=12030.

### Overall

| Метрика | Значение |
|---------|----------|
| Mean distance (m) | 2042.9 |
| Median distance (m) | 600.1 |
| P90 (m) | 2549.2 |
| P95 (m) | 8193.5 |

### По разрезам

- **Clean:** mean 2017 m, median 591 m, p90 2512 m.
- **Augmented:** mean 2069 m, median 610 m, p90 2582 m.
- **In overlap (с train):** mean 945 m, median 570 m, p90 1826 m.
- **No overlap:** mean 2194 m, median 605 m, p90 2737 m.

## Артефакты в этом каталоге

| Файл | Описание |
|------|----------|
| `config.json` | Конфиг эксперимента |
| `metrics.json` | Полные тестовые метрики и (при наличии) history |
| `error_histogram.png` | Гистограмма ошибок (м) на тесте |
| `scatter_errors.png` | Истинные vs предсказанные координаты (lat/lon) |

## Дополнительные артефакты (каталог эксперимента)

Исходный каталог прогона: `geocoding/experiments/run_2026-02-25_14-50-07/`.

- `best.pt` — веса лучшей эпохи.
- `test_predictions.json` — предсказания по всем тестовым примерам (address, true_lat/lon, pred_lat/lon, distance_m, is_augmented).
- `map_errors.html` — интерактивная карта: правильные (<100 м) и ошибки с отрезками истина→предсказание; подсветка и перелёт по клику/двойному клику.

## Сравнение с внешними геокодерами

Сравнение нашей модели с Dadata и Mistral на стратифицированной выборке (250 чистых + 250 аугментированных адресов) и карта сравнения:

- Отчёт и карта: `comparing/results/` — `comparison_report.json`, `comparison_summary.md`, `comparison_map.html`, `comparison_map_data.json`.
- Сводка сравнения включена в корневой отчёт: [EXPERIMENT_REPORT.md](../../EXPERIMENT_REPORT.md) в корне репозитория.
