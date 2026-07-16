# VK-LSVD: гибридные рекомендации с контентными эмбеддингами

Исследование влияния контентных эмбеддингов на качество рекомендаций  
на датасете [VK-LSVD](https://huggingface.co/datasets/deepvk/VK-LSVD)  
(сабсэмпл `ur0.01_ir0.01`: ~90k юзеров, ~125k айтемов, ~4M взаимодействий).

## Установка и данные

```bash
pip install -r requirements.txt
python scripts/download_data.py            # сабсэмпл (~26 МБ) + метаданные (~2.6 ГБ)
```

## Запуск эксперимента

Один запуск = один конфиг = один json в `results/`:

```bash
python -m src.train --config configs/popularity.yaml   # baseline
python -m src.train --config configs/ials.yaml         # iALS
```

По умолчанию оценка на validation (week 25). Test (week 26) трогаем
один раз, финальными моделями: `--eval-split test`.

## Структура

```
configs/          yaml-конфиги экспериментов
scripts/          скачивание данных, EDA, сборка таблиц и графиков
src/data/         загрузка, таргет (implicit feedback), id-маппинг
src/models/       popularity, ials, bpr, ease, knn, hybrid, two_tower
src/eval/         метрики (Recall/NDCG/MAP, coverage, novelty), cold/warm-срезы
src/train.py      единая точка входа
results/          json-результаты запусков
reports/          отчёт (docx, latex) и графики
```



## Результаты (test, неделя 26)


| Модель                  | Recall@50           | NDCG@10             | NDCG@10 cold        |
| ----------------------- | ------------------- | ------------------- | ------------------- |
| Popularity              | 0.0115              | 0.0000              | 0.0000              |
| Content kNN (d=64)      | 0.0087              | 0.0013              | 0.0010              |
| BPR-MF                  | 0.0539 ± 0.0004     | 0.0092 ± 0.0002     | 0.0000              |
| iALS                    | 0.0840 ± 0.0001     | **0.0118 ± 0.0000** | 0.0000              |
| iALS + content fallback | **0.0849 ± 0.0006** | 0.0116 ± 0.0001     | **0.0023 ± 0.0000** |


Главный вывод: контентные эмбеддинги почти не влияют на популярные айтемы,
но переводят cold-срез из нуля в измеримое качество; кривая качество-vs-d
насыщается после d=32. Полные результаты — в `results/`.

## Протокол оценки

- Global Temporal Split: train = недели 00–24, val = 25, test = 26 (как в датасете).
- Таргет: `like OR share OR bookmark OR watch_ratio >= 0.8` (+ 2 альтернативы для аблации).
- Метрики: Recall@10/50, NDCG@10/50, MAP@10; coverage@50, novelty@10.
- Срезы: cold/warm/popular айтемы, cold/active юзеры.

