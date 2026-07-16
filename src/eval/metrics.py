"""Метрики ранжирования для top-K рекомендаций."""

import numpy as np


def _per_user(recs_row: np.ndarray, pos: set, k: int) -> tuple[float, float, float]:
    topk = recs_row[:k]
    hits = np.fromiter((1.0 if i in pos else 0.0 for i in topk), dtype=np.float64, count=len(topk))
    n_pos = len(pos)
    recall = hits.sum() / n_pos

    discounts = 1.0 / np.log2(np.arange(2, len(topk) + 2))
    dcg = float((hits * discounts).sum())
    idcg = float(discounts[: min(n_pos, k)].sum())
    ndcg = dcg / idcg if idcg > 0 else 0.0

    cum = np.cumsum(hits)
    precision_at_hit = (cum * hits) / np.arange(1, len(topk) + 1)
    ap = float(precision_at_hit.sum() / min(n_pos, k))
    return recall, ndcg, ap


def evaluate_topk(recs: np.ndarray, positives: list[set],
                  ks: tuple[int, ...] = (10, 50)) -> dict[str, float]:
    acc: dict[str, list[float]] = {f"{m}@{k}": [] for k in ks for m in ("recall", "ndcg")}
    acc["map@10"] = []
    for row, pos in zip(recs, positives):
        if not pos:
            continue
        for k in ks:
            r, n, _ = _per_user(row, pos, k)
            acc[f"recall@{k}"].append(r)
            acc[f"ndcg@{k}"].append(n)
        _, _, ap = _per_user(row, pos, 10)
        acc["map@10"].append(ap)
    return {name: float(np.mean(vals)) if vals else 0.0 for name, vals in acc.items()}


def coverage(recs: np.ndarray, n_items: int, k: int = 50) -> float:
    return len(np.unique(recs[:, :k])) / n_items


def novelty(recs: np.ndarray, item_popularity: np.ndarray, k: int = 10) -> float:
    total = item_popularity.sum()
    p = np.maximum(item_popularity, 1) / max(total, 1)
    return float(-np.log2(p[recs[:, :k]]).mean())
