# OSM Address Dataset & Geocoding

Репозиторий содержит пайплайн сбора пар «адрес – координаты» из дампов OpenStreetMap и обучения модели геокодирования (адрес → lat, lon). Сбор данных выполняется локально, без облачных API и лимитов.

## Структура репозитория

| Каталог / файл | Назначение |
|----------------|------------|
| **scripts/** | Скачивание PBF и извлечение адресов в CSV из OSM-дампов |
| **geocoding/** | Обучение BERT-геокодера (адрес → координаты), метрики, визуализации |
| **eda_and_cleaning/** | EDA и очистка датасета (правила отсева, bbox, дедупликация) |
| **comparing/** | Сравнение обученной модели с Dadata и Mistral (метрики, оценка стоимости, пропуски) |
| **data/** | Каталог для CSV с адресами (исключён из git) |
| **Dockerfile.geocoding** | Образ для обучения геокодера с поддержкой SAGE (опечатки) |

## Требования

- [UV](https://docs.astral.sh/uv/)
- Python 3.12+

## Воспроизведение пайплайна

1. Установка зависимостей: `uv sync`.

2. Сбор датасета адресов из OSM (по умолчанию — Северо-Западный ФО, СПб):
   ```bash
   uv run python scripts/download_pbf.py
   uv run python scripts/fetch_osm_addresses.py data/northwestern-fed-district-latest.osm.pbf
   ```
   Выходной файл: `data/addresses_spb.csv`.

3. Опционально — очистка данных и формирование отчёта:
   ```bash
   uv run python eda_and_cleaning/analyze_addresses.py --csv data/addresses_spb.csv --out-dir eda_and_cleaning
   ```
   Очищенный датасет: `eda_and_cleaning/addresses_spb_cleaned.csv`.

4. Обучение геокодера:
   ```bash
   uv run python geocoding/train.py --csv eda_and_cleaning/addresses_spb_cleaned.csv
   ```
   Вариант с расширенными настройками (сплит 90/5/5, full finetune, 50 эпох): `bash geocoding/run_90_5_5_full50.sh`. Подробности — в [geocoding/README.md](geocoding/README.md).

5. Сравнение с альтернативами (Dadata, Mistral) выполняется скриптом в `comparing/`; вход — каталог эксперимента или `test_predictions.json`. См. [comparing/README.md](comparing/README.md).

## Сбор данных из OSM

Скачивание PBF: `uv run python scripts/download_pbf.py` (по умолчанию — Geofabrik Northwestern Fed District). Извлечение адресов: `uv run python scripts/fetch_osm_addresses.py [путь.pbf] -o ./data/addresses_spb.csv`. Доступны опции `--bbox`, `--json`; переменные окружения: `OSM_PBF_PATH`, `BBOX_SPB`, `OUTPUT_PATH`.

Источники PBF: [Geofabrik Russia](https://download.geofabrik.de/russia.html), [BBBike SanktPetersburg](https://download.bbbike.org/osm/bbbike/SanktPetersburg/).

## Docker

Обучение геокодера с SAGE выполняется в контейнере; данные и каталог экспериментов монтируются с хоста:
```bash
docker build -f Dockerfile.geocoding -t geocoding-train .
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/geocoding/experiments:/app/geocoding/experiments" geocoding-train
```
Подробнее — в [geocoding/README.md](geocoding/README.md#docker-обучение-с-sage).

## Документация по модулям

- [geocoding/README.md](geocoding/README.md) — обучение модели, выбор устройства (GPU/MPS/CPU), режимы, визуализации, Docker.
- [eda_and_cleaning/README.md](eda_and_cleaning/README.md) — правила очистки адресов и пайплайн EDA.
- [comparing/README.md](comparing/README.md) — сравнение с Dadata/Mistral, используемые API, прайсинг, справочно 2ГИС и Яндекс Карты.

## Лицензия

Данные OpenStreetMap распространяются под [ODbL](https://www.openstreetmap.org/copyright).
