# Geocoding: обучение BERT (адрес → координаты)

Модуль предназначен для обучения модели на базе **deepvk/USER2-small**, предсказывающей координаты (lat, lon) по тексту адреса.

## Запуск обучения

Из корня репозитория обучение запускается командой:
```bash
uv run python geocoding/train.py --csv data/addresses_spb.csv
```
Доступные опции: `--batch-size`, `--epochs`, `--lr`, `--patience`, `--max-length`, `--no-augment`, `--no-sage` и др. (полный список: `--help`).

Валидация строится только по адресам, не встречающимся в обучающей выборке (по нормализованному виду), чтобы early stopping и выбор лучшей модели не зависели от утечки. На тесте отдельно считаются метрики in_overlap / no_overlap.

## Устройство (CUDA, Apple Silicon, CPU)

Устройство выбирается автоматически в порядке: **CUDA** → **MPS** (Apple Silicon / Metal) → **CPU**. При локальном запуске на Mac с M1/M2/M3 используется MPS, если PyTorch собран с поддержкой MPS. Явное указание: `--device cuda`, `--device mps` или `--device cpu`. В Docker на Mac GPU в контейнер не передаётся; обучение на GPU/MPS выполняется локально, без Docker, при установленном при необходимости SAGE.

## Режимы обучения

- **`--head-only`** — обучается только голова (стабильный базовый режим).
- **`--full-finetune`** — все слои энкодера разморожены с первой эпохи.
- Без флагов — постепенная разморозка (gradual unfreeze). На MPS при разморозке возможны NaN; при превышении доли NaN (`--nan-stop-ratio`) обучение восстанавливает лучший чекпоинт и завершается.

Продолжение обучения с чекпоинта: `--resume путь/к/чекпоинту.pt --epochs N`. Оптимизер и расписание LR инициализируются заново; для дообучения обычно задаётся меньший `--lr` (например `1e-5`).

## Артефакты эксперимента

В каталоге `geocoding/experiments/run_YYYY-MM-DD_HH-MM-SS/` сохраняются:

- `config.json` — параметры запуска
- `metrics.json` — метрики по эпохам и на тесте (overall, in_overlap, no_overlap, clean, augmented)
- `train_log.txt` — лог обучения
- `history.png` — графики loss и метрик
- `best.pt`, `last.pt` — чекпоинты модели
- `test_predictions.json` — предсказания на тесте
- `map_errors.html` — интерактивная карта ошибок (точки по истинным координатам, цвет по величине ошибки; отрезки истинная→предсказанная)
- `error_histogram.png` — гистограмма ошибок (м)
- `scatter_errors.png` — scatter истинные vs предсказанные координаты

Визуализации строятся автоматически по окончании обучения (требуется `folium`). Отдельный пересчёт: `uv run python geocoding/visualize_test.py geocoding/experiments/run_YYYY-MM-DD_HH-MM-SS` (или путь к `test_predictions.json`). Опции: `--no-map`, `--no-hist`, `--no-scatter`, `--no-segments`, `--map-points`, `--out-dir`.

## Docker (обучение с SAGE)

Зависимости SAGE могут конфликтовать с версиями в PyPI, поэтому обучение с опечатками (SAGE) целесообразно запускать в контейнере:

```bash
docker build -f Dockerfile.geocoding -t geocoding-train .
docker run --rm -it \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/geocoding/experiments:/app/geocoding/experiments" \
  geocoding-train
```

Проверка на малом подмножестве (предварительно формируется `data/tiny_spb.csv`, например `head -101 data/addresses_spb.csv > data/tiny_spb.csv`):
```bash
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/geocoding/experiments:/app/geocoding/experiments" \
  geocoding-train python geocoding/train.py --csv data/tiny_spb.csv --epochs 1 --batch-size 8
```

В образ устанавливается [SAGE](https://github.com/ai-forever/sage) из GitHub; опечатки включаются по умолчанию, отключение — флагом `--no-sage`.

## Аугментации и SAGE

Опечатки (SAGE, CharAug) используются при установленном пакете. В Docker SAGE ставится при сборке образа. Локально установка: `pip install "git+https://github.com/ai-forever/sage.git"` (возможны конфликты версий). Дополнительно применяются перестановки частей адреса и добавление/удаление служебных слов.

## Зависимости

Основные: `torch`, `transformers`, `matplotlib`, `folium` (указаны в `pyproject.toml`). Для опечаток в Docker используется SAGE из GitHub (см. `Dockerfile.geocoding`).
