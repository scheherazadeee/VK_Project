# VK-LSVD: hybrid recommendations with content embeddings

A study of how content embeddings affect recommendation quality on the
[VK-LSVD](https://huggingface.co/datasets/deepvk/VK-LSVD) dataset
(subsample `ur0.01_ir0.01`: ~90k users, ~125k items, ~4M interactions).

## Setup and data

```bash
pip install -r requirements.txt
python scripts/download_data.py            # subsample (~26 MB) + metadata (~2.6 GB)
```

## Running an experiment

One run = one config = one json in `results/`:

```bash
python -m src.train --config configs/popularity.yaml   # baseline
python -m src.train --config configs/ials.yaml         # iALS
```

Evaluation is on validation (week 25) by default. The test week (26) is
touched once, with the final models: `--eval-split test`.

## Structure

```
configs/          yaml experiment configs
scripts/          data download, EDA, table and figure aggregation
src/data/         loading, target (implicit feedback), id mapping
src/models/       popularity, ials, bpr, ease, knn, hybrid, two_tower
src/eval/         metrics (Recall/NDCG/MAP, coverage, novelty), cold/warm slices
src/train.py      single entry point
results/          per-run json results
reports/          report (docx, latex) and figures
```

## Results (test, week 26)

| Model                   | Recall@50           | NDCG@10             | NDCG@10 cold        |
| ----------------------- | ------------------- | ------------------- | ------------------- |
| Popularity              | 0.0115              | 0.0000              | 0.0000              |
| Content kNN (d=64)      | 0.0087              | 0.0013              | 0.0010              |
| BPR-MF                  | 0.0539 ± 0.0004     | 0.0092 ± 0.0002     | 0.0000              |
| iALS                    | 0.0840 ± 0.0001     | **0.0118 ± 0.0000** | 0.0000              |
| iALS + content fallback | **0.0849 ± 0.0006** | 0.0116 ± 0.0001     | **0.0023 ± 0.0000** |

Main finding: content embeddings barely change quality on popular items, but
they move the cold slice from zero to measurable quality; the quality-vs-d
curve saturates past d=32. Full results are in `results/`.

## Evaluation protocol

- Global Temporal Split: train = weeks 00–24, validation = 25, test = 26 (as in the dataset).
- Target: `like OR share OR bookmark OR watch_ratio >= 0.8` (+ 2 alternatives for ablation).
- Metrics: Recall@10/50, NDCG@10/50, MAP@10; coverage@50, novelty@10.
- Slices: cold/warm/popular items, cold/active users.
