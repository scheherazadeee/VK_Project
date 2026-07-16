"""Срезы cold / warm / popular для анализа устойчивости."""

import numpy as np
import polars as pl

from .metrics import evaluate_topk

COLD_THRESHOLD = 5       # < 5 взаимодействий в train
POPULAR_QUANTILE = 0.99  # топ-1% айтемов по частоте


def item_slices(train_pos: pl.DataFrame, eval_item_ids: np.ndarray) -> dict[str, set]:
    counts = train_pos.group_by("item_id").len().rename({"len": "cnt"})
    eval_items = pl.DataFrame({"item_id": eval_item_ids}).join(counts, on="item_id", how="left")
    eval_items = eval_items.with_columns(pl.col("cnt").fill_null(0))

    pop_threshold = counts["cnt"].quantile(POPULAR_QUANTILE)
    cold = eval_items.filter(pl.col("cnt") < COLD_THRESHOLD)
    popular = eval_items.filter(pl.col("cnt") >= pop_threshold)
    warm = eval_items.filter((pl.col("cnt") >= COLD_THRESHOLD) & (pl.col("cnt") < pop_threshold))
    return {
        "cold_items": set(cold["item_id"].to_list()),
        "warm_items": set(warm["item_id"].to_list()),
        "popular_items": set(popular["item_id"].to_list()),
    }


def user_slices(train_pos: pl.DataFrame, eval_user_ids: np.ndarray) -> dict[str, set]:
    train_users = set(train_pos["user_id"].unique().to_list())
    all_users = set(eval_user_ids.tolist())
    return {
        "cold_users": all_users - train_users,
        "active_users": all_users & train_users,
    }


def evaluate_item_slices(recs: np.ndarray, positives: list[set],
                         slices: dict[str, set],
                         ks: tuple[int, ...] = (10, 50)) -> dict[str, dict[str, float]]:
    out = {}
    for name, items in slices.items():
        sliced = [pos & items for pos in positives]
        out[name] = evaluate_topk(recs, sliced, ks=ks)
    return out


def evaluate_user_slices(recs: np.ndarray, positives: list[set],
                         rec_user_ids: np.ndarray, slices: dict[str, set],
                         ks: tuple[int, ...] = (10, 50)) -> dict[str, dict[str, float]]:
    out = {}
    for name, users in slices.items():
        mask = np.isin(rec_user_ids, list(users))
        sliced_pos = [p for p, m in zip(positives, mask) if m]
        out[name] = evaluate_topk(recs[mask], sliced_pos, ks=ks)
    return out
