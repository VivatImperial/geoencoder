# OSM Address Dataset & Geocoding

Проект для сбора пар «адрес – координаты» из OpenStreetMap и обучения модели геокодирования (адрес → lat, lon). Всё локально: без облачных API и лимитов при сборе данных.

## Структура репозитория

| Каталог / файл | Описание |
|----------------|----------|
| **scripts/** | Скачивание PBF и извлечение адресов в CSV из OSM-дампов |
| **geocoding/** | Обучение BERT-геокодера (адрес → координаты), метрики, визуализации |
| **eda_and_cleaning/** | EDA и очистка датасета (правила отсева, bbox, дедупликация) |
| **comparing/** | Сравнение нашей модели с Dadata и Mistral (метрики, стоимость, пропуски) |
| **data/** | CSV с адресами (не в git: см. .gitignore) |
| **Dockerfile.geocoding** | Образ для обучения геокодера с SAGE (опечатки) |

## Требования

- [UV](https://docs.astral.sh/uv/)
- Python 3.12+

## Быстрый старт

1. Установить зависимости:
   ```bash
   uv sync
   ```

2. Собрать датасет адресов из OSM (по умолчанию — Северо-Западный ФО, СПб):
   ```bash
   uv run python scripts/download_pbf.py
   uv run python scripts/fetch_osm_addresses.py data/northwestern-fed-district-latest.osm.pbf
   ```
   Результат: `data/addresses_spb.csv`.

3. (Опционально) Очистить данные и получить отчёт:
   ```bash
   uv run python eda_and_cleaning/analyze_addresses.py --csv data/addresses_spb.csv --out-dir eda_and_cleaning
   ```
   Очищенный CSV: `eda_and_cleaning/addresses_spb_cleaned.csv`.

4. Обучить геокодер:
   ```bash
   uv run python geocoding/train.py --csv eda_and_cleaning/addresses_spb_cleaned.csv
   ```
   Или с полным набором опций (90/5/5, full finetune, 50 эпох): `bash geocoding/run_90_5_5_full50.sh`. Подробнее — в [geocoding/README.md](geocoding/README.md).

5. Сравнить с альтернативами (Dadata, Mistral):
   ```bash
   uv run python comparing/run_comparison.py --predictions geocoding/experiments/run_XXXX --out comparing/results --sample 500
   ```
   См. [comparing/README.md](comparing/README.md).

## Сбор данных из OSM (подробнее)

- Скачать PBF: `uv run python scripts/download_pbf.py` (по умолчанию Geofabrik Northwestern Fed District).
- Извлечь адреса: `uv run python scripts/fetch_osm_addresses.py [путь.pbf] -o ./data/addresses_spb.csv`.
- Опции: `--bbox`, `--json`. Переменные: `OSM_PBF_PATH`, `BBOX_SPB`, `OUTPUT_PATH`.

Источники PBF: [Geofabrik Russia](https://download.geofabrik.de/russia.html), [BBBike SanktPetersburg](https://download.bbbike.org/osm/bbbike/SanktPetersburg/).

## Docker (обучение геокодера с SAGE)

```bash
docker build -f Dockerfile.geocoding -t geocoding-train .
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/geocoding/experiments:/app/geocoding/experiments" geocoding-train
```

Данные и каталог экспериментов монтируются с хоста. Подробнее — в [geocoding/README.md](geocoding/README.md#docker-обучение-с-sage).

## Документация по частям

- [geocoding/README.md](geocoding/README.md) — обучение модели, GPU, режимы, визуализации, Docker.
- [eda_and_cleaning/README.md](eda_and_cleaning/README.md) — правила очистки адресов и пайплайн EDA.
- [comparing/README.md](comparing/README.md) — сравнение с Dadata/Mistral, API, прайсинг, 2ГИС/Яндекс.

## Лицензия

Данные OpenStreetMap — [ODbL](https://www.openstreetmap.org/copyright).
