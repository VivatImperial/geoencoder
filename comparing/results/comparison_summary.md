# Сравнение геокодеров

Тестовая выборка: 500 строк (из geocoding/experiments/run_2026-02-25_14-50-07/test_predictions.json).

## Метрики по разрезам (clean / augmented)

### our_model
- Пропуски: 0 из 500 (0.0%)
- Стоимость (оценка): 0 RUB — локальная модель
- **overall** (n=500): mean=1778 m, median=608 m, p90=2278 m
- **clean** (n=250): mean=1718 m, median=582 m, p90=2459 m
- **augmented** (n=250): mean=1838 m, median=632 m, p90=1912 m

### dadata
- Пропуски: 2 из 500 (0.4%)
- Стоимость (оценка): 100.0 RUB — 20 коп/запрос, см. dadata.ru/pricing
- **overall** (n=498): mean=11124 m, median=55 m, p90=36123 m
- **clean** (n=249): mean=13904 m, median=68 m, p90=37308 m
- **augmented** (n=249): mean=8344 m, median=49 m, p90=35654 m

### mistral
- Пропуски: 0 из 500 (0.0%)
- Стоимость (оценка): 0.012 USD — Mistral Small ~$0.05/1M in, $0.08/1M out
- **overall** (n=500): mean=16747 m, median=16566 m, p90=30663 m
- **clean** (n=250): mean=17318 m, median=17534 m, p90=31227 m
- **augmented** (n=250): mean=16176 m, median=16103 m, p90=30356 m
