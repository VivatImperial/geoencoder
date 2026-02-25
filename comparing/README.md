# Сравнение геокодера с альтернативами

Скрипт сравнивает обученную модель с Dadata и Mistral на тестовой выборке в двух разрезах: **аугментированные** и **неаугментированные** данные. В отчёт входят оценка стоимости прогона и число пропусков по каждому методу. Результаты записываются в JSON (`comparison_report.json`) и в Markdown (`comparison_summary.md`) в каталоге, заданном опцией `--out`.

## Методы в сравнении

| Подход | Описание | Пропуски |
|--------|----------|----------|
| **Our model** | Локальная BERT-модель (предсказания из `test_predictions.json`) | 0 |
| **Dadata** | Геокодирование через API (поиск по адресу, только РФ) | Частичное покрытие: СПб ~91% домов, qc_geo=5 — координаты не определены |
| **Mistral AI** | LLM по промпту (извлечение lat, lon без веб-поиска) | Зависит от контекста в модели |

## API и прайсинг (используемые в скрипте)

### Dadata

- Документация: https://dadata.ru/api/geocode/
- Метод: `POST https://cleaner.dadata.ru/api/v1/clean/address`, тело: JSON-массив строк `["адрес"]`.
- Ответ: `geo_lat`, `geo_lon`, `qc_geo` (0=дом … 5=не определены).
- Лимиты: 20 запросов/сек с одного IP.
- Стоимость: геокодирование не входит в подписку, 20 коп. за запрос (https://dadata.ru/pricing/ ).

### Mistral AI

- Документация: https://docs.mistral.ai/api/ (Chat Completion: `POST /v1/chat/completions`).
- Прайсинг: https://docs.mistral.ai/deployment/laplateforme/pricing/ , https://mistral.ai/pricing. Модель Mistral Small — порядка $0.05 / 1M input, $0.08 / 1M output. Короткий промпт и ответ «lat, lon» — примерно 400 input + 30 output токенов на адрес.

## Стоимость геокодирования: 2ГИС и Яндекс Карты (справочно)

В скрипте 2ГИС и Яндекс не вызываются; ниже — ориентировочные данные из открытых источников для сравнения рынка.

### 2ГИС

- Документация API: https://docs.2gis.com/ru/api/search/geocoder/overview
- Тарифы: открытого прайс-листа нет, стоимость определяется индивидуально (пакет запросов, лимиты, опции). Уточнение: https://platform.2gis.ru/ru/tariffs , заявка через dev.2gis.ru/api или отдел продаж.

### Яндекс Карты (Геокодер)

- Продукт: [Геокодер](https://yandex.ru/maps-api/products/geocoder-api) — конвертация адресов в координаты и обратно.
- Тарифы: https://yandex.ru/maps-api/tariffs (блок «JavaScript API + Геокодер» / ввод адресов). Стандартная лицензия: от 195 000 ₽/год (до 1 000 запросов/день) до 2 184 000 ₽/год (до 100 000 запросов/день). Сверх лимита: 390 ₽ за 1 000 запросов. Расширенная лицензия — выше. Бесплатно: до 1 000 запросов/день при соблюдении [условий](https://yandex.ru/dev/commercial/doc/ru/?from=mapsapi).

## Оценка стоимости одного полного прогона (N ≈ 12 000 адресов)

| Метод | Ориентир | Единица |
|-------|----------|--------|
| Our model | 0 | — |
| Dadata | ~2 400 ₽ | 20 коп/запрос |
| Mistral | ~0,3–0,5 $ | токены (Mistral Small) |

Ограничение числа запросов к API (для сокращения времени и затрат): опция `--sample 500`.

## Запуск скрипта

Ключи API задаются в файле `.env` в каталоге `comparing/`; шаблон — `comparing/.env.example` (DADATA_API_KEY, DADATA_SECRET_KEY, MISTRAL_API_KEY).

Запуск из корня репозитория:
```bash
uv run python comparing/run_comparison.py --predictions geocoding/experiments/run_XXXX --out comparing/results
```
С лимитом числа запросов к API:
```bash
uv run python comparing/run_comparison.py --predictions geocoding/experiments/run_XXXX --out comparing/results --sample 500
```

Выходные файлы: `comparing/results/comparison_report.json` (полная сводка по методам: overall/clean/augmented, n_misses, cost_estimate) и `comparing/results/comparison_summary.md` (краткая текстовая сводка).
