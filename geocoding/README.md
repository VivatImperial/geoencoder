# Geocoding: обучение BERT (адрес → координаты)

Обучение модели на базе **deepvk/USER2-small** для предсказания координат (lat, lon) по тексту адреса.

## Запуск

```bash
# из корня репозитория
uv run python geocoding/train.py --csv data/addresses_spb.csv
```

Опции: `--batch-size`, `--epochs`, `--lr`, `--patience`, `--max-length`, `--no-augment`, `--no-sage` и др. (см. `--help`).

**Валидация и тест:** валидация строится только по «незнакомым» адресам (нормализованный адрес не встречается в train), чтобы early stopping и выбор лучшей модели были честными. В тесте по-прежнему считаются отдельные метрики in_overlap / no_overlap.

## GPU (CUDA и Apple Silicon)

Устройство выбирается автоматически: **CUDA** → **MPS** (Apple Silicon / Metal) → **CPU**. На Mac с M1/M2/M3 при локальном запуске (`uv run python geocoding/train.py ...`) будет использоваться MPS, если PyTorch собран с поддержкой MPS. Принудительно задать устройство: `--device cuda`, `--device mps` или `--device cpu`.

В Docker на Mac GPU в контейнер не пробрасывается, там всегда CPU. Чтобы гонять обучение на ядрах Mac, запускай скрипт локально (без Docker), с установленным SAGE при необходимости.

**Режимы обучения:** **`--head-only`** — только голова (стабильный базовый вариант). **`--full-finetune`** — все слои энкодера разморожены с первой эпохи. Без флагов — постепенная разморозка (gradual unfreeze). На MPS при разморозке возможны NaN; при массовых NaN обучение восстановит best и остановится (`--nan-stop-ratio`).

**Продолжить обучение:** загрузить веса из чекпоинта и дообучить ещё N эпох: `--resume путь/к/чекпоинту.pt --epochs N`. Оптимизер и расписание LR стартуют заново; для дообучения часто берут меньший `--lr` (например `1e-5`).

## Результаты

В `geocoding/experiments/run_YYYY-MM-DD_HH-MM-SS/` сохраняются:

- `config.json` — параметры запуска
- `metrics.json` — метрики по эпохам и тесту (overall, in_overlap, no_overlap, clean, augmented)
- `train_log.txt` — лог обучения
- `history.png` — графики loss и метрик
- `best.pt`, `last.pt` — чекпоинты модели
- `test_predictions.json` — предсказания на тесте (для визуализаций)
- `map_errors.html` — интерактивная карта ошибок (точки по истинным координатам, цвет по величине ошибки; синие отрезки — истинная→предсказанная)
- `error_histogram.png` — гистограмма ошибок (м)
- `scatter_errors.png` — scatter истинные vs предсказанные координаты (широта и долгота)

После обучения визуализации строятся автоматически (нужен `folium`). Отдельно пересобрать: `uv run python geocoding/visualize_test.py geocoding/experiments/run_YYYY-MM-DD_HH-MM-SS` (или путь к `test_predictions.json`). Опции: `--no-map`, `--no-hist`, `--no-scatter`, `--no-segments`, `--map-points 2000`, `--out-dir ...`.

## Docker (обучение с SAGE)

Зависимости SAGE конфликтуют с текущими версиями в PyPI, поэтому обучение с опечатками (SAGE) удобно запускать в контейнере:

```bash
# Сборка (в корне репозитория)
docker build -f Dockerfile.geocoding -t geocoding-train .

# Запуск: монтируем data и каталог экспериментов
docker run --rm -it \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/geocoding/experiments:/app/geocoding/experiments" \
  geocoding-train

# Быстрая проверка на 100 строках (предварительно: head -101 data/addresses_spb.csv > data/tiny_spb.csv)
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/geocoding/experiments:/app/geocoding/experiments" \
  geocoding-train python geocoding/train.py --csv data/tiny_spb.csv --epochs 1 --batch-size 8
```

В образе уже установлен [SAGE](https://github.com/ai-forever/sage) из GitHub; опечатки включаются автоматически (флаг `--no-sage` отключает их при необходимости).

## Аугментации и SAGE

Опечатки через SAGE (CharAug) используются, если пакет установлен. В Docker SAGE ставится при сборке образа. Локально без Docker можно поставить вручную: `pip install "git+https://github.com/ai-forever/sage.git"` (возможны конфликты версий с проектом). Дополнительно всегда работают перестановки частей адреса и добавление/удаление мусорных слов.

## Зависимости

Основные: `torch`, `transformers`, `matplotlib`, `folium` (в `pyproject.toml`). Для опечаток в Docker используется SAGE из GitHub (см. `Dockerfile.geocoding`).
